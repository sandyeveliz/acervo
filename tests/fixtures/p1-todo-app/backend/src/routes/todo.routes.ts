import { Router } from "express";
import { listTodos, getTodo, createTodo, updateTodo, deleteTodo, getStats } from "../controllers/todo.controller";
import { requireAuth } from "../middleware/auth.middleware";

const router = Router();

router.use(requireAuth);

router.get("/", listTodos);
router.get("/stats", getStats);
router.get("/:id", getTodo);
router.post("/", createTodo);
router.put("/:id", updateTodo);
router.delete("/:id", deleteTodo);

export default router;
