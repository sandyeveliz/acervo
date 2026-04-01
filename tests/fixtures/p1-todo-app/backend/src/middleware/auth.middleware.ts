import { Request, Response, NextFunction } from "express";
import { AuthService } from "../services/auth.service";

declare global {
  namespace Express {
    interface Request {
      userId?: number;
    }
  }
}

const authService = new AuthService();

export function requireAuth(req: Request, res: Response, next: NextFunction): void {
  const header = req.headers.authorization;

  if (!header || !header.startsWith("Bearer ")) {
    res.status(401).json({ error: "Authorization header required" });
    return;
  }

  const token = header.slice(7);

  try {
    const { userId } = authService.verifyToken(token);
    req.userId = userId;
    next();
  } catch {
    res.status(401).json({ error: "Invalid or expired token" });
  }
}
