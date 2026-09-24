import baseline from "./index";
import dense from "./dense";
import in10 from "./in10";
import cub from "./cub";
export { In10Control, In10Artifact } from "./in10";

// Original routes/secrets/allowlist are unchanged. The isolated dense prefix
// inherits the same IP gate and admits only its own project and two run IDs.
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/in10/') || url.pathname.startsWith('/k96coco/')) return in10.fetch(request, env);
    if (url.pathname.startsWith('/cub-v1/')) {
      url.pathname=url.pathname.slice('/cub-v1'.length);
      return cub.fetch(new Request(url,request),env);
    }
    if (url.pathname.startsWith("/dense-v2/")) {
      url.pathname = url.pathname.slice("/dense-v2".length);
      return dense.fetch(new Request(url, request), env);
    }
    return baseline.fetch(request, env);
  },
} satisfies ExportedHandler<Env>;
