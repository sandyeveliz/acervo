export const config = {
  port: parseInt(process.env.PORT || "3001", 10),
  jwtSecret: process.env.JWT_SECRET || "dev-secret-change-in-production",
  jwtExpiresIn: process.env.JWT_EXPIRES_IN || "24h",
  corsOrigin: process.env.CORS_ORIGIN || "http://localhost:5173",
  nodeEnv: process.env.NODE_ENV || "development",
  dbPath: process.env.DB_PATH || "./data/todos.db",
} as const;

export type Config = typeof config;
