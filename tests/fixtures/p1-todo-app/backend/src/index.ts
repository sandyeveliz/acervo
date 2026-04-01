import app from "./app";
import { config } from "./config/env";
import { closeDatabase } from "./config/database";

const server = app.listen(config.port, () => {
  console.log(`Server running on port ${config.port} (${config.nodeEnv})`);
});

process.on("SIGTERM", () => {
  console.log("SIGTERM received, shutting down...");
  server.close(() => {
    closeDatabase();
    process.exit(0);
  });
});
