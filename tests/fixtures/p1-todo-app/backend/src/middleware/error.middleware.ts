import { Request, Response, NextFunction } from "express";

export function errorHandler(err: Error, _req: Request, res: Response, _next: NextFunction): void {
  console.error(`[ERROR] ${err.name}: ${err.message}`);

  if (err.name === "SyntaxError" && "body" in err) {
    res.status(400).json({ error: "Invalid JSON in request body" });
    return;
  }

  res.status(500).json({
    error: process.env.NODE_ENV === "production" ? "Internal server error" : err.message,
  });
}

export function notFoundHandler(_req: Request, res: Response): void {
  res.status(404).json({ error: "Route not found" });
}
