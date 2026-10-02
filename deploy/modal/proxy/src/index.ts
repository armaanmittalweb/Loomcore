// loomcore-api.amittal.dev: forwards the console's requests to the runtime on Modal and tells it
// who the visitor is (Workers' own subrequests all leave from Cloudflare's addresses, so the
// runtime's per-visitor rate limit needs the address passed on, vouched for by a shared key).
interface Env {
  ORIGIN: string
  PROXY_KEY: string
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url)
    const headers = new Headers(request.headers)
    headers.delete('host')
    headers.set('x-client-ip', request.headers.get('cf-connecting-ip') ?? '')
    headers.set('x-proxy-key', env.PROXY_KEY)
    return fetch(new URL(url.pathname + url.search, env.ORIGIN), {
      method: request.method,
      headers,
      body: request.method === 'GET' || request.method === 'HEAD' ? undefined : request.body,
      redirect: 'manual',
    })
  },
}
