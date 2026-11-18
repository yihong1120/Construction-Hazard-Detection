// Dependency-free single request to an isolated Chrome DevTools target.
const [url, method, parameters] = process.argv.slice(2);
const ws = new WebSocket(url);
const timer = setTimeout(() => { console.error('CDP timeout'); process.exit(1); }, 45000);
ws.onopen = () => ws.send(JSON.stringify({ id: 1, method, params: JSON.parse(parameters) }));
ws.onmessage = event => {
  const response = JSON.parse(event.data);
  if (response.id !== 1) return;
  console.log(JSON.stringify(response)); clearTimeout(timer); ws.close();
};
ws.onerror = () => { console.error('CDP socket failed'); process.exit(1); };
