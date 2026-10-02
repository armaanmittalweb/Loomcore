# Deploying the runtime on Oracle Cloud (Always Free, Arm)

The runtime server (`space/Dockerfile`, `space/server.py`) runs on an Oracle Cloud Always Free
Ampere A1 VM (Neoverse N1, aarch64, 4 OCPU / 24 GB, Ubuntu 24.04), always on, bound to
`127.0.0.1:7860` and published only through a Cloudflare Tunnel at
`https://loomcore-api.amittal.dev`. Nothing on the VM listens on a public port.

## 1. The VM

Create an instance: shape `VM.Standard.A1.Flex`, 4 OCPU, 24 GB, image Ubuntu 24.04 (aarch64),
with your SSH key. No ingress rules beyond SSH are needed (the tunnel only dials out).

```
ssh ubuntu@<vm-ip>
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y ca-certificates curl git
# Docker Engine + compose plugin (Docker's own repository; arm64 is supported)
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ubuntu && newgrp docker
```

## 2. Build and run

```
git clone https://github.com/armaanmittalweb/loomcore.git && cd loomcore
docker compose -f deploy/oracle/compose.yaml up -d --build      # LOOMCORE_REF=main ... to build another ref
docker compose -f deploy/oracle/compose.yaml logs -f             # the build's ctest and pytest output is in the build log
curl -s 127.0.0.1:7860/health
```

The build clones the repository at `LOOMCORE_REF` (default `live`) inside the image, prepares the
models (CPU torch, first stage only), builds Loomcore with CMake + Ninja and fails unless `ctest`,
the binding tests and the server tests all pass. Expect 20 to 40 minutes the first time on 4 OCPU;
later builds reuse the cached stages. For a private repository, put a read-only token in
`deploy/oracle/github_token.txt` and uncomment the two `secrets` blocks in `compose.yaml`.

The container restarts on failure and on boot (`restart: always`, with Docker enabled at boot by
the package). At start it loads the graph, warms both precisions, then runs `loomcore_bench` once
(`GET /bench`), which takes a few seconds.

To update: `git pull && docker compose -f deploy/oracle/compose.yaml up -d --build`.

## 3. Cloudflare Tunnel

```
curl -L -o cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
cloudflared tunnel login                         # authorise the amittal.dev zone
cloudflared tunnel create loomcore
cloudflared tunnel route dns loomcore loomcore-api.amittal.dev
```

`~/.cloudflared/config.yml` (use the tunnel id and credentials file `create` printed):

```
tunnel: <tunnel-id>
credentials-file: /home/ubuntu/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: loomcore-api.amittal.dev
    service: http://127.0.0.1:7860
  - service: http_status:404
```

Then run it as a service and check it from anywhere:

```
sudo cloudflared --config /home/ubuntu/.cloudflared/config.yml service install
sudo systemctl enable --now cloudflared
curl -s https://loomcore-api.amittal.dev/health
```

The host is one subdomain level (`loomcore-api.amittal.dev`), so Cloudflare's universal
certificate covers it. The server's CORS already allows `https://loomcore.amittal.dev`; its rate
limit reads Cloudflare's `CF-Connecting-IP`.

## 4. After it is up

From a checkout on any machine:

```
cd web
npm run e2e:live -- --space https://loomcore-api.amittal.dev      # drives the real runtime through the console
npm run record -- --from https://loomcore-api.amittal.dev         # replace the recorded runs with ones from the VM
```

Commit the new `web/src/recorded/*.json` and redeploy the console: it then opens on runs
recorded on the live runtime, labelled as such.
