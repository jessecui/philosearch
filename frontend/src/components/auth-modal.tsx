import { useEffect, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Loader2 } from "lucide-react";
import Cookies from "js-cookie";

import { LogoMark } from "@/components/logo-mark";
import { API_BASE, errorMessage } from "@/lib/api";

interface AuthModalProps {
  onAuthenticated: () => void;
}

export function AuthModal({ onAuthenticated }: AuthModalProps) {
  const [answer, setAnswer] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // The input is disabled while verifying, which drops focus; give it back
  // so a mistyped name can be retyped straight away
  useEffect(() => {
    if (!isLoading) inputRef.current?.focus();
  }, [isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    let response: Response;
    try {
      response = await fetch(`${API_BASE}/validate-creator`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ answer }),
      });
    } catch {
      setError("Network error. Please try again.");
      setIsLoading(false);
      return;
    }

    try {
      if (!response.ok) {
        throw new Error(await errorMessage(response));
      }

      const data = await response.json();

      if (data.valid) {
        // UI hint only, so the prompt isn't shown again on reload. Access is
        // checked by the server's HttpOnly session cookie set in this response.
        Cookies.set("creator_auth", "true", { expires: 1 });
        onAuthenticated();
      } else {
        setError("That name doesn't match our records. Please try again.");
        setAnswer("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={true} modal>
      <DialogContent
        className="sm:max-w-md"
        onInteractOutside={(e) => e.preventDefault()}
        onEscapeKeyDown={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-lg font-semibold text-gray-900">
            <LogoMark />
            PhiloSearch
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <DialogDescription className="text-sm text-gray-500">
              A semantic search engine for philosophical texts
            </DialogDescription>
            <label
              htmlFor="creator-question"
              className="block pt-2 text-sm font-medium text-gray-900"
            >
              To continue, enter the name of this project's creator.
            </label>
            <input
              ref={inputRef}
              id="creator-question"
              name="creator-question"
              type="text"
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              disabled={isLoading}
              autoComplete="one-time-code"
              className="w-full h-10 px-3 rounded-md border border-gray-300 bg-white text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20 disabled:bg-gray-50"
              placeholder="Full name"
              autoFocus
            />
          </div>
          {error && (
            <p role="alert" className="text-sm text-red-700">{error}</p>
          )}
          <button
            type="submit"
            disabled={isLoading || !answer.trim()}
            className="w-full h-10 rounded-md bg-blue-700 text-white text-sm font-medium hover:bg-blue-800 disabled:bg-gray-200 disabled:text-gray-500 disabled:cursor-not-allowed flex items-center justify-center gap-2 cursor-pointer"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Verifying…
              </>
            ) : (
              "Continue"
            )}
          </button>
        </form>
      </DialogContent>
    </Dialog>
  );
}
