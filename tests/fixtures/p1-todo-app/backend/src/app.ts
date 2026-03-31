import express from "express";
import cors from "cors";
import todoRoutes from "./routes/todo.routes";
import authRoutes from "./routes/auth.routes";
import { errorHandler, notFoundHandler } from "./middleware/error.middleware";
import { config } from "./config/env";

const app = express();

app.use(cors({ origin: config.corsOrigin }));
app.use(express.json());

// Health check
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/todos", todoRoutes);

// Error handling
app.use(notFoundHandler);
app.use(errorHandler);

export default app;
