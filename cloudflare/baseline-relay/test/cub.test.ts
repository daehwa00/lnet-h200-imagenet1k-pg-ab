import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import cub from '../src/cub';
import { CAMPAIGN } from '../src/cub.generated';
const env = { WANDB_API_KEY:'test-only', ALLOWED_EGRESS_IPS:'test-egress',
  RELAY_RATE_LIMITER:{limit:async()=>({success:true})}} as Env;
function request(run=Object.keys(CAMPAIGN.runsById)[0],project=CAMPAIGN.project as string,ip='test-egress') {
  return new Request(`https://test/files/daehwa/${project}/${run}/file_stream`,{
    method:'POST',headers:{'Content-Type':'application/json','CF-Connecting-IP':ip},
    body:JSON.stringify({files:{'wandb-history.jsonl':{offset:0,content:['{"epoch":1}']}}})});
}
beforeEach(()=>vi.stubGlobal('fetch',vi.fn(async()=>Response.json({data:{}}))));
afterEach(()=>vi.unstubAllGlobals());
it('admits exactly 17 CUB telemetry streams',async()=>{
  expect(Object.keys(CAMPAIGN.runsById)).toHaveLength(17);
  for(const id of Object.keys(CAMPAIGN.runsById)) expect((await cub.fetch(request(id),env)).status).toBe(200);
});
it('rejects other runs, projects and egress',async()=>{
  expect((await cub.fetch(request('other'),env)).status).toBe(403);
  expect((await cub.fetch(request(undefined,'alphabet2d-imagenet1k-10pct'),env)).status).toBe(403);
  expect((await cub.fetch(request(undefined,undefined,'other'),env)).status).toBe(403);
  expect(fetch).not.toHaveBeenCalled();
});
