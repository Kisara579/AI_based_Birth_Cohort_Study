import { useEffect, useState } from "react";
import { apiGet } from "../../api/client.js";

export default function ApiStatusDot() {
  const [ok, setOk] = useState(null); // null = unknown

  useEffect(() => {
    let cancelled = false;

    async function ping() {
      try {
        const data = await apiGet("/health");
        if (!cancelled) setOk(Boolean(data?.ok));
      } catch {
        if (!cancelled) setOk(false);
      }
    }

    ping();
    const t = setInterval(ping, 5000);

    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  const title =
    ok === null ? "API: checking..." : ok ? "API: online" : "API: offline";

  return <span className={`apiDot ${ok ? "apiDotOk" : "apiDotBad"}`} title={title} />;
}
