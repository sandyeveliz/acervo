import { describe, it, expect, beforeEach } from "vitest";
import { TodoService, ValidationError } from "../src/services/todo.service";

describe("TodoService", () => {
  const service = new TodoService();
  const testUserId = 1;

  describe("createTodo", () => {
    it("should reject empty title", async () => {
      await expect(service.createTodo(testUserId, { title: "" })).rejects.toThrow(ValidationError);
    });

    it("should reject title over 200 characters", async () => {
      const longTitle = "x".repeat(201);
      await expect(service.createTodo(testUserId, { title: longTitle })).rejects.toThrow(ValidationError);
    });

    it("should reject invalid due date", async () => {
      await expect(
        service.createTodo(testUserId, { title: "Test", due_date: "not-a-date" })
      ).rejects.toThrow(ValidationError);
    });

    it("should trim title whitespace", async () => {
      const todo = await service.createTodo(testUserId, { title: "  Buy groceries  " });
      expect(todo.title).toBe("Buy groceries");
    });

    it("should default priority to medium", async () => {
      const todo = await service.createTodo(testUserId, { title: "Test" });
      expect(todo.priority).toBe("medium");
    });
  });

  describe("getStats", () => {
    it("should return zero completion rate for no todos", async () => {
      const stats = await service.getStats(999);
      expect(stats.completionRate).toBe(0);
      expect(stats.total).toBe(0);
    });
  });
});
