import { useState, useCallback, useEffect } from "react";
import type { AuthUser } from "../types/todo";
import { login as apiLogin, register as apiRegister, logout as apiLogout, isAuthenticated } from "../services/api";

interface UseAuthResult {
  user: AuthUser | null;
  authenticated: boolean;
  loading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
}

export function useAuth(): UseAuthResult {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const authenticated = isAuthenticated();

  useEffect(() => {
    const stored = localStorage.getItem("auth_user");
    if (stored && authenticated) {
      setUser(JSON.parse(stored));
    }
  }, [authenticated]);

  const login = useCallback(async (email: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiLogin(email, password);
      setUser(result.user);
      localStorage.setItem("auth_user", JSON.stringify(result.user));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const register = useCallback(async (email: string, password: string, name: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiRegister(email, password, name);
      setUser(result.user);
      localStorage.setItem("auth_user", JSON.stringify(result.user));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    apiLogout();
    setUser(null);
    localStorage.removeItem("auth_user");
  }, []);

  return { user, authenticated, loading, error, login, register, logout };
}
