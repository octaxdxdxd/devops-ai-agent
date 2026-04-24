import { EventEmitter } from "node:events";
import type { RunTraceEvent } from "../shared/schemas.js";

export class RunBus {
  private readonly emitter = new EventEmitter();

  publish(event: RunTraceEvent): void {
    this.emitter.emit(event.runId, event);
  }

  subscribe(runId: string, listener: (event: RunTraceEvent) => void): () => void {
    this.emitter.on(runId, listener);
    return () => this.emitter.off(runId, listener);
  }
}
