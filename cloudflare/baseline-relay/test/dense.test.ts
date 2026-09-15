import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import worker from "../src/entry";
import { CAMPAIGN as BASE } from "../src/campaign.generated";
import { CAMPAIGN as DENSE } from "../src/dense.generated";

const env: Env = { WANDB_API_KEY: "test-only-not-a-real-key", ALLOWED_EGRESS_IPS: "test-egress",
  RELAY_RATE_LIMITER: { limit: async () => ({success:true}) } };
const run = Object.keys(DENSE.runsById)[0];
function request(project: string=DENSE.project, id: string=run, ip: string="test-egress"): Request {
  return new Request(`https://relay.invalid/dense-v2/files/${DENSE.entity}/${project}/${id}/file_stream`, {
    method:"POST", headers:{"Content-Type":"application/json","CF-Connecting-IP":ip},
    body:JSON.stringify({files:{"wandb-history.jsonl":{offset:0,content:['{"optimizer_updates":20}']}}}),
  });
}
describe("additive dense relay scope", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn(async()=>Response.json({data:{}}))));
  afterEach(()=>vi.unstubAllGlobals());
  it("keeps original health unchanged",async()=>{
    const r=await worker.fetch(new Request("https://relay.invalid/healthz"),env);
    expect(await r.json()).toMatchObject({campaign_id:BASE.campaignId,manifest_sha256:BASE.manifestSha256});
  });
  it("exposes only the separate non-secret dense health",async()=>{
    const r=await worker.fetch(new Request("https://relay.invalid/dense-v2/healthz"),env);
    expect(await r.json()).toMatchObject({campaign_id:DENSE.campaignId,manifest_sha256:DENSE.manifestSha256});
  });
  it("forwards an allowed dense stream without leaking the route prefix",async()=>{
    expect((await worker.fetch(request(),env)).status).toBe(200);
    expect(fetch).toHaveBeenCalledWith(`https://api.wandb.ai/files/${DENSE.entity}/${DENSE.project}/${run}/file_stream`,expect.anything());
  });
  it("preserves IP restriction",async()=>{
    expect((await worker.fetch(request(DENSE.project,run,"untrusted-egress"),env)).status).toBe(403);
    expect(fetch).not.toHaveBeenCalled();
  });
  it("rejects unregistered run IDs",async()=>{
    expect((await worker.fetch(request(DENSE.project,"not-registered"),env)).status).toBe(403);
    expect(fetch).not.toHaveBeenCalled();
  });
  it("rejects cross-project writes",async()=>{
    expect((await worker.fetch(request(BASE.project),env)).status).toBe(403);
    expect(fetch).not.toHaveBeenCalled();
  });
});
