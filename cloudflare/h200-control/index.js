import { DurableObject } from 'cloudflare:workers';

export class Campaign extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    ctx.storage.sql.exec('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)');
    ctx.storage.sql.exec('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT)');
  }
  control() {
    const rows = this.ctx.storage.sql.exec("SELECT value FROM state WHERE key='stop'").toArray();
    return {stop: rows.length ? JSON.parse(rows[0].value) : false};
  }
  stop() {
    this.ctx.storage.sql.exec("INSERT OR REPLACE INTO state VALUES ('stop', 'true')");
    return this.control();
  }
  append(value) {
    this.ctx.storage.sql.exec('INSERT INTO events(value) VALUES (?)', JSON.stringify(value));
    // Bounded remote tail; full logs remain on the persistent H200 volume.
    this.ctx.storage.sql.exec('DELETE FROM events WHERE id < (SELECT MAX(id)-2000 FROM events)');
    return {ok:true};
  }
  logs(after) {
    return this.ctx.storage.sql.exec('SELECT id,value FROM events WHERE id>? ORDER BY id LIMIT 100', after).toArray();
  }
}

export default {
  async fetch(req, env) {
    const auth = req.headers.get('Authorization');
    const owner = !!env.OWNER_TOKEN && auth === `Bearer ${env.OWNER_TOKEN}`;
    const agent = !!env.AGENT_TOKEN && auth === `Bearer ${env.AGENT_TOKEN}`;
    if (!owner && !agent) return new Response('Unauthorized', {status:401});
    const url = new URL(req.url);
    const campaign = env.CAMPAIGN.getByName(url.searchParams.get('test')==='1' ? 'isolated-control-test-v1' : 'k96-coco-501-509-521-v1');
    let value;
    if (req.method==='GET' && url.pathname==='/control') value=await campaign.control();
    else if (owner && req.method==='POST' && url.pathname==='/stop') value=await campaign.stop();
    else if (owner && req.method==='GET' && url.pathname==='/logs') {
      const after=Number(url.searchParams.get('after') || 0);
      if (!Number.isSafeInteger(after) || after<0) return new Response('Bad cursor',{status:400});
      value=await campaign.logs(after);
    } else if (agent && req.method==='POST' && url.pathname==='/events') {
      const text=await req.text();
      if (text.length>64000) return new Response('Too large',{status:413});
      let body;
      try {body=JSON.parse(text);} catch {return new Response('Bad JSON',{status:400});}
      if (!Number.isSafeInteger(body.offset) || body.offset<0 || typeof body.text!=='string' || body.text.length>24000 || typeof body.status!=='object') return new Response('Bad event',{status:400});
      value=await campaign.append(body);
    } else return new Response('Forbidden',{status:403});
    return Response.json(value,{headers:{'Cache-Control':'no-store'}});
  }
};
