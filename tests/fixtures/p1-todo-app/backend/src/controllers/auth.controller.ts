import { Request, Response, NextFunction } from "express";
import { AuthService, AuthError } from "../services/auth.service";

const authService = new AuthService();

export async function register(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const { email, password, name } = req.body;

    if (!email || !password || !name) {
      res.status(400).json({ error: "Email, password, and name are required" });
      return;
    }

    const result = await authService.register(email, password, name);
    res.status(201).json({ data: result });
  } catch (err) {
    if (err instanceof AuthError) {
      res.status(400).json({ error: err.message });
      return;
    }
    next(err);
  }
}

export async function login(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      res.status(400).json({ error: "Email and password are required" });
      return;
    }

    const result = await authService.login(email, password);
    res.json({ data: result });
  } catch (err) {
    if (err instanceof AuthError) {
      res.status(401).json({ error: err.message });
      return;
    }
    next(err);
  }
}

export async function me(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const userId = req.userId!;
    const user = await import("../models/user.model").then(m => m.findById(userId));

    if (!user) {
      res.status(404).json({ error: "User not found" });
      return;
    }

    res.json({
      data: { id: user.id, email: user.email, name: user.name, created_at: user.created_at },
    });
  } catch (err) {
    next(err);
  }
}
