// Backend base URL. In production the backend serves the frontend, so requests
// go to the same origin (""). In `npm run dev`, requests go through the Vite
// proxy at /api (see vite.config.ts). Both keep the session cookie same-origin,
// which the app requires: the cookie is SameSite=Lax and these requests don't
// send credentials cross-origin. Override with VITE_API_URL only for a path on
// the same site (e.g. a reverse proxy prefix).
export const API_BASE: string =
  import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "/api" : "");

// Read FastAPI's {"detail": "..."} error body, falling back to the status code.
export async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    // Not JSON; use the status below
  }
  return `Request failed (HTTP ${response.status})`;
}
