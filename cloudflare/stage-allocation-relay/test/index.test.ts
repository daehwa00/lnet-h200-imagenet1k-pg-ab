import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CAMPAIGN } from "../src/campaign.generated";
import worker from "../src/index";

const PROBE_QUERY = `query ProbeServerCapabilities {
  QueryType: __type(name: "Query") {
    ...fieldData
  }
  MutationType: __type(name: "Mutation") {
    ...fieldData
  }
  ServerInfoType: __type(name: "ServerInfo") {
    ...fieldData
  }
}

fragment fieldData on __Type {
  fields {
    name
  }
}
`;

const UPSERT_QUERY = `
mutation UpsertBucket ($id: String, $name: String, $project: String, $entity: String, $groupName: String, $description: String, $displayName: String, $notes: String, $commit: String, $config: JSONString, $host: String, $debug: Boolean, $program: String, $repo: String, $jobType: String, $state: String, $sweep: String, $tags: [String!], $summaryMetrics: JSONString) {
\tupsertBucket(input: {id:$id,name:$name,groupName:$groupName,modelName:$project,entityName:$entity,description:$description,displayName:$displayName,notes:$notes,config:$config,commit:$commit,host:$host,debug:$debug,jobProgram:$program,jobRepo:$repo,jobType:$jobType,state:$state,sweep:$sweep,tags:$tags,summaryMetrics:$summaryMetrics}) {
\t\tbucket {
\t\t\tid
\t\t\tname
\t\t\tdisplayName
\t\t\tdescription
\t\t\tconfig
\t\t\tsweepName
\t\t\tproject {
\t\t\t\tid
\t\t\t\tname
\t\t\t\tentity {
\t\t\t\t\tid
\t\t\t\t\tname
\t\t\t\t}
\t\t\t}
\t\t\thistoryLineCount
\t\t}
\t\tinserted
\t}
}
`;

function testEnv(rateAllowed = true): Env {
  return {
    ALLOWED_EGRESS_IPS: "test-egress",
    RELAY_RATE_LIMITER: {
      limit: async () => ({ success: rateAllowed }),
    },
    WANDB_API_KEY: "test-only-not-a-real-key",
  };
}

function post(path: string, body: unknown, ip = "test-egress"): Request {
  return new Request(`https://relay.invalid${path}`, {
    body: JSON.stringify(body),
    headers: {
      "CF-Connecting-IP": ip,
      "Content-Type": "application/json",
    },
    method: "POST",
  });
}

function lastForwardedPayload(): { variables: Record<string, unknown> } {
  const call = vi.mocked(fetch).mock.calls.at(-1);
  const body = call?.[1]?.body as ArrayBuffer;
  return JSON.parse(new TextDecoder().decode(body));
}

describe("H200 W&B relay", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ data: {} })));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("publishes a versioned, non-secret health contract", async () => {
    const response = await worker.fetch(new Request("https://relay.invalid/healthz"), testEnv());
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({
      authorization_manifest_sha256: CAMPAIGN.authorizationManifestSha256,
      campaign_id: CAMPAIGN.campaignId,
      manifest_sha256: CAMPAIGN.manifestSha256,
      ok: true,
      protocol_version: CAMPAIGN.protocolVersion,
    });
  });

  it("forwards an exact traced GraphQL operation", async () => {
    const response = await worker.fetch(post("/graphql", {
      query: PROBE_QUERY,
      variables: {},
    }), testEnv());
    expect(response.status).toBe(200);
    expect(response.headers.get("X-Relay-Protocol")).toBe(CAMPAIGN.protocolVersion);
  });

  it("accepts the exact traced UpsertBucket shape for a frozen run", async () => {
    const entry = Object.entries(CAMPAIGN.runsById)
      .find(([, run]) => run.group === CAMPAIGN.group);
    expect(entry).toBeDefined();
    const [runId, run] = entry!;
    const response = await worker.fetch(post("/graphql", {
      operationName: "UpsertBucket",
      query: UPSERT_QUERY,
      variables: {
        commit: null,
        config: "{}",
        debug: false,
        description: null,
        displayName: run.displayName,
        entity: CAMPAIGN.entity,
        groupName: run.group,
        host: "untrusted-host-is-overwritten",
        id: null,
        jobType: null,
        name: runId,
        notes: null,
        program: "untrusted-program-is-overwritten",
        project: run.project,
        repo: null,
        state: "running",
        summaryMetrics: null,
        sweep: null,
        tags: [...run.tags],
      },
    }), testEnv());
    expect(response.status).toBe(200);
    expect(lastForwardedPayload().variables.program).toBe(run.program);
  });

  it("accepts an allowlisted supplemental campaign run", async () => {
    const entry = Object.entries(CAMPAIGN.runsById)
      .find(([, run]) => run.group !== CAMPAIGN.group);
    expect(entry).toBeDefined();
    const [runId, run] = entry!;
    const response = await worker.fetch(post("/graphql", {
      operationName: "UpsertBucket",
      query: UPSERT_QUERY,
      variables: {
        commit: null,
        config: "{}",
        debug: false,
        description: null,
        displayName: run.displayName,
        entity: CAMPAIGN.entity,
        groupName: run.group,
        host: "untrusted-host-is-overwritten",
        id: null,
        jobType: null,
        name: runId,
        notes: null,
        program: "untrusted-program-is-overwritten",
        project: run.project,
        repo: null,
        state: "running",
        summaryMetrics: null,
        sweep: null,
        tags: [...run.tags],
      },
    }), testEnv());
    expect(response.status).toBe(200);
    expect(lastForwardedPayload().variables.program).toBe(run.program);
  });

  it("accepts both frozen K64 P-depth-interaction training runs", async () => {
    for (const runId of ["ae6ce4374b8ea076", "6bdea5b0c2a0ee6a"] as const) {
      const run = CAMPAIGN.runsById[runId];
      expect(run.group).toBe("R2K3-K64-PDepthInteraction-H200-S501");
      const response = await worker.fetch(post("/graphql", {
        operationName: "UpsertBucket",
        query: UPSERT_QUERY,
        variables: {
          commit: null,
          config: "{}",
          debug: false,
          description: null,
          displayName: run.displayName,
          entity: CAMPAIGN.entity,
          groupName: run.group,
          host: null,
          id: null,
          jobType: null,
          name: runId,
          notes: null,
          program: null,
          project: run.project,
          repo: null,
          state: "running",
          summaryMetrics: null,
          sweep: null,
          tags: [...run.tags],
        },
      }), testEnv());
      expect(response.status).toBe(200);
      expect(lastForwardedPayload().variables.program).toBe(run.program);
    }
  });

  it("accepts the frozen K64 campaign in its canonical non-H200 project", async () => {
    const entry = Object.entries(CAMPAIGN.runsById)
      .find(([, run]) => run.project !== CAMPAIGN.project);
    expect(entry).toBeDefined();
    const [runId, run] = entry!;
    const response = await worker.fetch(post("/graphql", {
      operationName: "UpsertBucket",
      query: UPSERT_QUERY,
      variables: {
        commit: null,
        config: "{}",
        debug: false,
        description: null,
        displayName: run.displayName,
        entity: CAMPAIGN.entity,
        groupName: run.group,
        host: null,
        id: null,
        jobType: null,
        name: runId,
        notes: null,
        program: null,
        project: run.project,
        repo: null,
        state: "running",
        summaryMetrics: null,
        sweep: null,
        tags: [...run.tags],
      },
    }), testEnv());
    expect(response.status).toBe(200);
  });

  it("rejects a request outside the secret egress allowlist", async () => {
    const response = await worker.fetch(post("/graphql", {}, "other-egress"), testEnv());
    expect(response.status).toBe(403);
  });

  it("rejects untraced GraphQL text", async () => {
    const response = await worker.fetch(post("/graphql", {
      operationName: "ProbeServerCapabilities",
      query: "query ProbeServerCapabilities { __typename }",
      variables: {},
    }), testEnv());
    expect(response.status).toBe(403);
  });

  it("rejects console output from file_stream", async () => {
    const runId = Object.keys(CAMPAIGN.runsById)[0];
    const response = await worker.fetch(post(
      `/files/${CAMPAIGN.entity}/${CAMPAIGN.project}/${runId}/file_stream`,
      { files: { "output.log": { content: ["must not pass"], offset: 0 } } },
    ), testEnv());
    expect(response.status).toBe(403);
  });

  it("forwards bounded wandb-history file_stream data", async () => {
    const runId = Object.keys(CAMPAIGN.runsById)[0];
    const path = `/files/${CAMPAIGN.entity}/${CAMPAIGN.project}/${runId}/file_stream`;
    const response = await worker.fetch(post(path, {
      files: { "wandb-history.jsonl": { content: ["{\"epoch\":1}"], offset: 0 } },
    }), testEnv());
    expect(response.status).toBe(200);
  });

  it("forwards the exact empty W&B heartbeat", async () => {
    const runId = Object.keys(CAMPAIGN.runsById)[0];
    const path = `/files/${CAMPAIGN.entity}/${CAMPAIGN.project}/${runId}/file_stream`;
    const response = await worker.fetch(post(path, {}), testEnv());

    expect(response.status).toBe(200);
  });

  it("forwards an allowed uploaded-file acknowledgement", async () => {
    const runId = Object.keys(CAMPAIGN.runsById)[0];
    const path = `/files/${CAMPAIGN.entity}/${CAMPAIGN.project}/${runId}/file_stream`;
    const response = await worker.fetch(post(path, {
      uploaded: ["config.yaml"],
    }), testEnv());

    expect(response.status).toBe(200);
  });

  it("passes upstream stop feedback back to the SDK", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ stopped: false })));
    const runId = Object.keys(CAMPAIGN.runsById)[0];
    const path = `/files/${CAMPAIGN.entity}/${CAMPAIGN.project}/${runId}/file_stream`;
    const response = await worker.fetch(post(path, {}), testEnv());

    await expect(response.json()).resolves.toEqual({ stopped: false });
  });

  it("accepts a signed int32 exit code with a completed run", async () => {
    const runId = Object.keys(CAMPAIGN.runsById)[0];
    const path = `/files/${CAMPAIGN.entity}/${CAMPAIGN.project}/${runId}/file_stream`;
    const response = await worker.fetch(post(path, {
      complete: true,
      exitcode: -15,
    }), testEnv());

    expect(response.status).toBe(200);
  });

  it("rejects an uploaded-file acknowledgement outside the campaign", async () => {
    const runId = Object.keys(CAMPAIGN.runsById)[0];
    const path = `/files/${CAMPAIGN.entity}/${CAMPAIGN.project}/${runId}/file_stream`;
    const response = await worker.fetch(post(path, {
      uploaded: ["requirements.txt"],
    }), testEnv());

    expect(response.status).toBe(403);
  });

  it("enforces the per-run rate limiter", async () => {
    const response = await worker.fetch(post("/graphql", {
      query: PROBE_QUERY,
      variables: {},
    }), testEnv(false));
    expect(response.status).toBe(429);
  });

  it("fails closed when Worker secrets are missing", async () => {
    const env = testEnv();
    env.WANDB_API_KEY = "";
    const response = await worker.fetch(post("/graphql", {}), env);
    expect(response.status).toBe(503);
  });
});
