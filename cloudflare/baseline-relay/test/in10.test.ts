import {env,runInDurableObject} from 'cloudflare:test';
import {describe,it,expect,beforeEach} from 'vitest';
import service from '../src/in10';

const token='a'.repeat(64),owner='private-test-owner';
const bindings={...env,IN10_OWNER_TOKEN:owner,ALLOWED_EGRESS_IPS:'198.51.100.10',RELAY_RATE_LIMITER:{limit:async()=>({success:true})}} as unknown as Env;
async function hash(v:string){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(v)))).map(x=>x.toString(16).padStart(2,'0')).join('');}
async function call(path:string,key=token,session='',value?:unknown,ip='198.51.100.10'){
 return service.fetch(new Request('https://test/in10'+path,{method:value===undefined?'GET':'POST',
  headers:{'Authorization':'Bearer '+key,'X-Session-ID':session,'CF-Connecting-IP':ip,'Content-Type':'application/json'},
  body:value===undefined?undefined:JSON.stringify(value)}),bindings);
}
async function enroll(){const r=await call('/enroll',token,'',{pod:'job-daehwa00-900-test',code_sha:'b'.repeat(40),token_hash:await hash(token)});expect(r.status).toBe(200);return (await r.json() as any).session_id as string;}

describe('IN10 isolated control and private checkpoint handoff',()=>{
 beforeEach(async()=>{
  await runInDurableObject(env.IN10_CONTROL.getByName('simclr-in1k10-v1'),(_instance,state)=>{
   state.storage.sql.exec('DELETE FROM meta');state.storage.sql.exec('DELETE FROM events');
  });
  await runInDurableObject(env.IN10_ARTIFACT.getByName('v1/va_k96-501'),(_instance,state)=>{
   state.storage.sql.exec('DELETE FROM chunks');state.storage.sql.exec('DELETE FROM versions');
  });
 });
 it('rejects enrollment outside the original IP gate',async()=>{
  expect((await call('/enroll',token,'',{},'203.0.113.9')).status).toBe(403);
 });
 it('keeps agent read/ingest separate from owner stop/ready',async()=>{
  const id=await enroll();
  expect((await call('/control','c'.repeat(64),id)).status).toBe(401);
  expect((await call('/command',token,id,{action:'stop'})).status).toBe(403);
  expect((await call('/command',owner,id,{action:'ready',job:'va_k96-501'})).status).toBe(200);
  await call('/command',owner,id,{action:'stop'});
  const c=await (await call('/control',token,id)).json() as any;
  expect(c.stop).toBe(true);expect(c.ready_jobs).toEqual(['va_k96-501']);
 });
 it('makes a lost enrollment response retry safe',async()=>{
  const id=await enroll();expect(await enroll()).toBe(id);
 });
 it('round trips JSON encoded checkpoint chunks',async()=>{
  const id=await enroll(),sha=await hash('binary'),base='/artifact/va_k96-501/';
  const data=JSON.stringify({base64:btoa('binary')});
  await call(base+'begin',token,id,{sha,bytes:6,chunks:1,epoch:1});
  const r=await service.fetch(new Request('https://test/in10'+base+sha+'/0',{
   method:'PUT',headers:{'Authorization':'Bearer '+token,'X-Session-ID':id,
    'CF-Connecting-IP':'198.51.100.10','Content-Type':'application/json','Content-Length':String(data.length)},body:data}),bindings);
  expect(r.status).toBe(200);
  const get=await service.fetch(new Request('https://test/in10'+base+sha+'/0',{
   headers:{'Authorization':'Bearer '+owner,'Accept':'application/json'}}),bindings);
  expect(await get.json()).toEqual({base64:btoa('binary')});
 });
 it('deduplicates event retries and does not expose agent hash to the observer',async()=>{
  const id=await enroll();const event={seq:0,logs:'hello',status:{stage:'test'}};
  await call('/events',token,id,event);await call('/events',token,id,event);
  const view=await (await call('/snapshot',owner)).json() as any;
  expect(view.events.length).toBe(1);expect(view.session.tokenHash).toBeUndefined();
 });
 it('requires all checkpoint chunks and owner verification',async()=>{
  const id=await enroll(),sha=await hash('probe');const base='/artifact/va_k96-501/';
  await call(base+'begin',token,id,{sha,bytes:5,chunks:1,epoch:1,workers:0});
  expect((await call(base+sha+'/finish',token,id,{})).status).toBe(409);
  const put=await service.fetch(new Request('https://test/in10'+base+sha+'/0',{method:'PUT',
   headers:{'Authorization':'Bearer '+token,'X-Session-ID':id,'CF-Connecting-IP':'198.51.100.10','Content-Length':'5'},body:'probe'}),bindings);
  expect(put.status).toBe(200);
  await call(base+sha+'/finish',token,id,{});
  expect((await (await call(base+'latest',owner)).json() as any).verified).toBe(false);
  expect((await call(base+sha+'/ack',token,id,{})).status).toBe(403);
  await call(base+sha+'/ack',owner,id,{});
  expect((await (await call(base+'latest',owner)).json() as any).verified).toBe(true);
  expect(await (await call(base+sha+'/0',owner)).text()).toBe('probe');
 });
});
