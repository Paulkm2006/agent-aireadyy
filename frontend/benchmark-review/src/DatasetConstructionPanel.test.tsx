/* @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { DatasetConstructionPanel } from "./DatasetConstructionPanel";

afterEach(() => cleanup());

describe("DatasetConstructionPanel", () => {
  it("shows the complete ten-protocol product workflow", () => {
    render(<DatasetConstructionPanel batchOutputDir="C:\\batch-42" taskType="denovo" />);

    expect(screen.getByTestId("dataset-construction-panel")).toBeTruthy();
    const batchInput = screen.getByLabelText("Batch 输出目录") as HTMLInputElement;
    expect(batchInput.value).toContain("batch-42");
    expect(screen.getByRole("button", { name: "构建十类拆分" })).toBeTruthy();
    expect(screen.getByText(/十类可审计/)).toBeTruthy();
    expect(screen.getByLabelText("构建配置")).toBeTruthy();
  });

  it("normalizes legacy AI-ready task names at the shared product boundary", () => {
    render(<DatasetConstructionPanel taskType="rt_prediction" />);

    const task = screen.getByLabelText("模型任务") as HTMLSelectElement;
    expect(task.value).toBe("rt_prediction");
  });
});
