import { CAMPAIGN } from "./campaign.generated";

type JsonObject = Record<string, unknown>;
type OperationName = keyof typeof CAMPAIGN.graphqlOperations;
type RunId = keyof typeof CAMPAIGN.runsById;
type GraphQLPayload = {
  operationName?: unknown;
  query?: unknown;
  variables?: unknown;
};

const JSON_CONTENT_TYPE = /^application\/json(?:\s*;\s*charset=utf-8)?$/i;
const NO_VARIABLE_OPERATIONS = new Set<string>(CAMPAIGN.noVariableOperations);
const RUN_FILES = new Set<string>(CAMPAIGN.runFiles);
const STREAM_FILES = new Set<string>(CAMPAIGN.streamFiles);
const ALLOWED_STATES = new Set<unknown>(CAMPAIGN.allowedStates);

function isObject(value: unknown): value is JsonObject {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function splitSet(value: string): Set<string> {
  return new Set(value.split(",").map((item) => item.trim()).filter(Boolean));
}

function hasExactKeys(value: JsonObject, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return actual.length === sortedExpected.length
    && actual.every((key, index) => key === sortedExpected[index]);
}

function isRunId(value: unknown): value is RunId {
  return typeof value === "string" && Object.hasOwn(CAMPAIGN.runsById, value);
}

function operationForHash(hash: string): OperationName | undefined {
  for (const operation of Object.keys(CAMPAIGN.graphqlOperations) as OperationName[]) {
    if (CAMPAIGN.graphqlOperations[operation] === hash) return operation;
  }
  return undefined;
}

async function sha256(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function readBoundedBody(request: Request, limit: number): Promise<Uint8Array | null> {
  const rawLength = request.headers.get("Content-Length");
  if (rawLength !== null) {
    if (!/^\d+$/.test(rawLength) || Number(rawLength) > limit) return null;
  }
  if (!request.body) return new Uint8Array();

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > limit) {
      await reader.cancel("relay body limit exceeded");
      return null;
    }
    chunks.push(value);
  }
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

function parseJson(body: Uint8Array): unknown | undefined {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true, ignoreBOM: false }).decode(body));
  } catch {
    return undefined;
  }
}

function validateVariables(operation: OperationName, variables: unknown): variables is JsonObject {
  const value = variables ?? {};
  if (!isObject(value)) return false;
  if (NO_VARIABLE_OPERATIONS.has(operation)) return Object.keys(value).length === 0;

  if (operation === "RunResumeStatus") {
    return hasExactKeys(value, ["entity", "name", "project"])
      && value.entity === CAMPAIGN.entity
      && value.project === CAMPAIGN.project
      && isRunId(value.name);
  }
  if (operation === "RunStoppedStatus") {
    return hasExactKeys(value, ["entityName", "projectName", "runId"])
      && value.entityName === CAMPAIGN.entity
      && value.projectName === CAMPAIGN.project
      && isRunId(value.runId);
  }
  if (operation === "CreateRunFiles") {
    if (!hasExactKeys(value, ["entity", "files", "project", "run"])) return false;
    if (value.entity !== CAMPAIGN.entity || value.project !== CAMPAIGN.project || !isRunId(value.run)) return false;
    if (!Array.isArray(value.files) || value.files.length < 1 || value.files.length > RUN_FILES.size) return false;
    return new Set(value.files).size === value.files.length
      && value.files.every((file) => typeof file === "string" && RUN_FILES.has(file));
  }
  if (operation === "UpsertBucket") {
    if (!hasExactKeys(value, CAMPAIGN.upsertBucketVariableKeys)) return false;
    if (value.entity !== CAMPAIGN.entity || value.project !== CAMPAIGN.project || !isRunId(value.name)) return false;
    const run = CAMPAIGN.runsById[value.name];
    if (value.groupName !== CAMPAIGN.group || value.displayName !== run.displayName) return false;
    if (!Array.isArray(value.tags) || value.tags.length !== run.tags.length) return false;
    if (!value.tags.every((tag, index) => tag === run.tags[index])) return false;
    for (const key of ["config", "summaryMetrics"] as const) {
      if (value[key] !== null && typeof value[key] !== "string") return false;
      if (typeof value[key] === "string" && value[key].length > CAMPAIGN.maxGraphqlBodyBytes) return false;
    }
    return ALLOWED_STATES.has(value.state);
  }
  return false;
}

function runIdFor(operation: OperationName, variables: JsonObject): RunId | undefined {
  const candidates = [variables.name, variables.run, variables.runId];
  return candidates.find(isRunId);
}

function sanitizeVariables(operation: OperationName, variables: JsonObject): JsonObject {
  if (operation !== "UpsertBucket") return variables;
  return {
    ...variables,
    commit: null,
    debug: false,
    description: null,
    host: "h200-managed",
    id: null,
    jobType: null,
    notes: null,
    program: "h200/run_baselines.sh",
    repo: "https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab",
    sweep: null,
  };
}

function validateStreamFile(value: unknown): boolean {
  if (!isObject(value) || !hasExactKeys(value, ["content", "offset"])) return false;
  if (!Number.isSafeInteger(value.offset) || (value.offset as number) < 0) return false;
  if (!Array.isArray(value.content) || value.content.length > 1024) return false;
  return value.content.every((item) => typeof item === "string" && item.length <= 1024 * 1024);
}

function validateFileStreamBody(payload: unknown): payload is JsonObject {
  if (!isObject(payload)) return false;
  const allowedKeys = new Set(["complete", "dropped", "exitcode", "failed", "files", "preempting", "uploaded"]);
  if (Object.keys(payload).some((key) => !allowedKeys.has(key))) return false;
  if (payload.files !== undefined) {
    if (!isObject(payload.files)) return false;
    for (const [name, value] of Object.entries(payload.files)) {
      if (!STREAM_FILES.has(name) || !validateStreamFile(value)) return false;
    }
  }
  if (payload.uploaded !== undefined) {
    if (!Array.isArray(payload.uploaded) || payload.uploaded.length > RUN_FILES.size) return false;
    if (payload.uploaded.some((name) => typeof name !== "string" || !RUN_FILES.has(name))) return false;
  }
  for (const key of ["complete", "failed", "preempting"] as const) {
    if (payload[key] !== undefined && typeof payload[key] !== "boolean") return false;
  }
  if (payload.dropped !== undefined && (!Number.isSafeInteger(payload.dropped) || (payload.dropped as number) < 0)) return false;
  if (payload.exitcode !== undefined && payload.exitcode !== 0 && payload.exitcode !== 1) return false;
  return payload.files !== undefined || payload.complete !== undefined || payload.preempting !== undefined;
}

function basicAuthorization(key: string): string {
  return `Basic ${btoa(`api:${key}`)}`;
}

function logEvent(level: "info" | "warn" | "error", fields: JsonObject): void {
  const payload = JSON.stringify({
    campaign_id: CAMPAIGN.campaignId,
    protocol_version: CAMPAIGN.protocolVersion,
    ...fields,
  });
  if (level === "error") console.error(payload);
  else if (level === "warn") console.warn(payload);
  else console.log(payload);
}

function reject(reason: string, status: number, requestId: string, route: string): Response {
  logEvent("warn", { event: "relay_reject", reason, request_id: requestId, route, status });
  return Response.json({ error: "request rejected", request_id: requestId }, {
    status,
    headers: { "Cache-Control": "no-store" },
  });
}

async function forward(path: string, body: Uint8Array, apiKey: string, requestId: string): Promise<Response> {
  try {
    const upstream = await fetch(`${CAMPAIGN.upstreamOrigin}${path}`, {
      method: "POST",
      headers: {
        Authorization: basicAuthorization(apiKey),
        "Content-Type": "application/json",
        "User-Agent": `lnet-h200-wandb-relay/${CAMPAIGN.protocolVersion}`,
      },
      body: body.buffer as ArrayBuffer,
      redirect: "manual",
    });
    if (upstream.status >= 300 && upstream.status < 400) {
      return reject("upstream_redirect", 502, requestId, "upstream");
    }
    logEvent("info", { event: "relay_forward", request_id: requestId, status: upstream.status });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Cache-Control": "no-store",
        "Content-Type": upstream.headers.get("Content-Type") ?? "application/json",
        "X-Relay-Protocol": CAMPAIGN.protocolVersion,
      },
    });
  } catch (error) {
    logEvent("error", {
      error_type: error instanceof Error ? error.name : "UnknownError",
      event: "relay_upstream_error",
      request_id: requestId,
    });
    return Response.json({ error: "upstream unavailable", request_id: requestId }, {
      status: 502,
      headers: { "Cache-Control": "no-store" },
    });
  }
}

function syntheticViewer(): Response {
  return Response.json({
    data: {
      viewer: {
        entity: CAMPAIGN.entity,
        flags: "{}",
        id: "lnet-h200-relay",
        teams: { edges: [] },
        username: CAMPAIGN.entity,
      },
    },
  }, { headers: { "Cache-Control": "no-store" } });
}

async function enforceRateLimit(env: Env, key: string): Promise<boolean> {
  const result = await env.RELAY_RATE_LIMITER.limit({ key: `${CAMPAIGN.protocolVersion}:${key}` });
  return result.success;
}

async function handleRequest(request: Request, env: Env): Promise<Response> {
  const requestId = crypto.randomUUID();
  const url = new URL(request.url);
  if (request.method === "GET" && url.pathname === "/healthz" && !url.search) {
    return Response.json({
      campaign_id: CAMPAIGN.campaignId,
      manifest_sha256: CAMPAIGN.manifestSha256,
      ok: true,
      protocol_version: CAMPAIGN.protocolVersion,
      sdk_version: CAMPAIGN.sdkVersion,
    }, { headers: { "Cache-Control": "no-store" } });
  }

  const sourceIp = request.headers.get("CF-Connecting-IP");
  if (!env.WANDB_API_KEY || !env.ALLOWED_EGRESS_IPS) {
    return reject("relay_not_configured", 503, requestId, "preflight");
  }
  if (!sourceIp || !splitSet(env.ALLOWED_EGRESS_IPS).has(sourceIp)) {
    return reject("source_not_allowed", 403, requestId, "preflight");
  }
  if (request.method !== "POST" || url.search || url.pathname.includes("%") || url.pathname.includes("\\")) {
    return reject("route_not_allowed", 404, requestId, "preflight");
  }
  if (!JSON_CONTENT_TYPE.test(request.headers.get("Content-Type") ?? "")) {
    return reject("content_type_not_allowed", 415, requestId, "preflight");
  }
  if (request.headers.has("Content-Encoding") && request.headers.get("Content-Encoding") !== "identity") {
    return reject("content_encoding_not_allowed", 415, requestId, "preflight");
  }

  const streamMatch = /^\/files\/([^/]+)\/([^/]+)\/([^/]+)\/file_stream$/.exec(url.pathname);
  if (streamMatch) {
    const [, pathEntity, pathProject, pathRun] = streamMatch;
    if (pathEntity !== CAMPAIGN.entity || pathProject !== CAMPAIGN.project || !isRunId(pathRun)) {
      return reject("stream_scope_mismatch", 403, requestId, "file_stream");
    }
    const body = await readBoundedBody(request, CAMPAIGN.maxFileStreamBodyBytes);
    if (!body) return reject("body_too_large", 413, requestId, "file_stream");
    const payload = parseJson(body);
    if (payload === undefined) return reject("invalid_json", 400, requestId, "file_stream");
    if (!validateFileStreamBody(payload)) return reject("stream_shape_not_allowed", 403, requestId, "file_stream");
    if (!await enforceRateLimit(env, pathRun)) return reject("rate_limited", 429, requestId, "file_stream");
    return forward(url.pathname, body, env.WANDB_API_KEY, requestId);
  }

  if (url.pathname !== "/graphql") return reject("route_not_allowed", 404, requestId, "preflight");
  const body = await readBoundedBody(request, CAMPAIGN.maxGraphqlBodyBytes);
  if (!body) return reject("body_too_large", 413, requestId, "graphql");
  const parsed = parseJson(body);
  if (!isObject(parsed) || typeof parsed.query !== "string") {
    return reject("graphql_envelope_not_allowed", 403, requestId, "graphql");
  }
  const payload: GraphQLPayload = parsed;
  const query = parsed.query;
  const operation = operationForHash(await sha256(query));
  if (!operation) {
    return reject("graphql_operation_not_allowed", 403, requestId, "graphql");
  }
  if (!hasExactKeys(parsed, CAMPAIGN.graphqlEnvelopeKeys[operation])) {
    return reject("graphql_envelope_not_allowed", 403, requestId, "graphql");
  }
  if (payload.operationName !== undefined && payload.operationName !== operation) {
    return reject("graphql_operation_not_allowed", 403, requestId, "graphql");
  }
  if (!validateVariables(operation, payload.variables)) {
    return reject("graphql_variables_not_allowed", 403, requestId, "graphql");
  }
  const variables = (payload.variables ?? {}) as JsonObject;
  const runId = runIdFor(operation, variables);
  if (!await enforceRateLimit(env, runId ?? "metadata")) {
    return reject("rate_limited", 429, requestId, "graphql");
  }
  if (operation === "Viewer") return syntheticViewer();

  const sanitized = new TextEncoder().encode(JSON.stringify({
    operationName: operation,
    query,
    variables: sanitizeVariables(operation, variables),
  }));
  return forward("/graphql", sanitized, env.WANDB_API_KEY, requestId);
}

export default {
  fetch: handleRequest,
} satisfies ExportedHandler<Env>;
