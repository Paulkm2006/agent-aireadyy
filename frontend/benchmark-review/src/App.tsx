import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Button,
  Content,
  Header,
  HeaderMenuButton,
  HeaderName,
  InlineLoading,
  SideNav,
  SideNavDivider,
  SideNavItems,
  SideNavLink,
  SkipToContent,
  Tag,
  Tile,
} from "@carbon/react";

import { buildInfoLabel, buildInfoTitle } from "./build-info";
import type { GrillExternalCommand } from "./CarbonAgentChat";
import { DiscoveryContextRail } from "./DiscoveryContextRail";
import { OperationsConsole, type OperationsTab } from "./OperationsConsole";
import { createEmptyIntent, type GrillPhase, type IntentSpec } from "./intent-spec";
import { reconcileSearchTermSelection, searchTermCandidates } from "./grill-tree";
import type {
  DiscoveryBatchHandoff,
  DiscoveryJob,
  WorkflowRecord,
} from "./workflow-api";

const CarbonAgentChat = lazy(() =>
  import("./CarbonAgentChat").then((module) => ({
    default: module.CarbonAgentChat,
  })),
);
const AiReadyPanel = lazy(() =>
  import("./AiReadyPanel").then((module) => ({
    default: module.AiReadyPanel,
  })),
);
const BatchPanel = lazy(() =>
  import("./BatchPanel").then((module) => ({
    default: module.BatchPanel,
  })),
);
const OperationsHistory = lazy(() =>
  import("./OperationsHistory").then((module) => ({
    default: module.OperationsHistory,
  })),
);
const SettingsPanel = lazy(() =>
  import("./SettingsPanel").then((module) => ({
    default: module.SettingsPanel,
  })),
);
const SingleTaskPanel = lazy(() =>
  import("./SingleTaskPanel").then((module) => ({
    default: module.SingleTaskPanel,
  })),
);

type ProductSection =
  | "current"
  | "history"
  | "batch"
  | "single"
  | "ai-ready"
  | "settings";

const CURRENT_DISCOVERY_STORAGE_KEY = "pride.discovery.activeJobId";
const productSections = new Set<ProductSection>([
  "current",
  "history",
  "batch",
  "single",
  "ai-ready",
  "settings",
]);

const sectionFromLocation = (): ProductSection => {
  const candidate = window.location.hash.replace(/^#/, "") as ProductSection;
  return productSections.has(candidate) ? candidate : "current";
};

const navItems: Array<{ id: ProductSection; label: string }> = [
  { id: "current", label: "当前任务" },
  { id: "history", label: "历史任务" },
  { id: "batch", label: "批量处理" },
  { id: "single", label: "单文件处理" },
  { id: "ai-ready", label: "AI-ready 构建" },
  { id: "settings", label: "系统设置" },
];

const sectionFromLegacyIndex = (index: number): ProductSection => {
  const sections: Record<number, ProductSection> = {
    0: "current",
    1: "single",
    2: "batch",
    3: "ai-ready",
    4: "history",
    5: "settings",
  };
  return sections[index] || "current";
};

export default function App() {
  const [section, setSection] = useState<ProductSection>(sectionFromLocation);
  const [currentWorkspaceMounted, setCurrentWorkspaceMounted] = useState(
    () => sectionFromLocation() === "current",
  );
  const [navExpanded, setNavExpanded] = useState(false);
  const [currentJobId, setCurrentJobId] = useState(
    () => window.localStorage.getItem(CURRENT_DISCOVERY_STORAGE_KEY) || "",
  );
  const currentJobIdRef = useRef(currentJobId);
  const [operationsTab, setOperationsTab] =
    useState<OperationsTab>("overview");
  const [legacyJob, setLegacyJob] = useState<DiscoveryJob | null>(null);
  const [taskId, setTaskId] = useState("");
  const [batchId, setBatchId] = useState("");
  const [batchSeedInputs, setBatchSeedInputs] = useState("");
  const [batchSeedRecords, setBatchSeedRecords] = useState<WorkflowRecord[]>([]);
  const [batchSeedSource, setBatchSeedSource] = useState<{
    jobId?: string;
    discoveryId?: string;
    batchIndex?: number;
  }>({});
  const [batchSeedToken, setBatchSeedToken] = useState(0);
  const [intent, setIntent] = useState<IntentSpec>(() => createEmptyIntent());
  const [selectedSearchTerms, setSelectedSearchTerms] = useState<string[]>([]);
  const [phase, setPhase] = useState<GrillPhase>("idle");
  const [externalCommand, setExternalCommand] =
    useState<GrillExternalCommand | null>(null);
  const [resultOpenRequest, setResultOpenRequest] = useState<{
    jobId: string;
    token: number;
  } | null>(null);
  const searchTermCandidatesForIntent = useMemo(
    () => searchTermCandidates(intent),
    [intent],
  );
  const searchTermCandidateKey = searchTermCandidatesForIntent.join("\u0000");

  useEffect(() => {
    setSelectedSearchTerms((current) =>
      reconcileSearchTermSelection(current, intent),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchTermCandidateKey]);

  useEffect(() => {
    const synchronize = () => setSection(sectionFromLocation());
    window.addEventListener("hashchange", synchronize);
    window.addEventListener("popstate", synchronize);
    return () => {
      window.removeEventListener("hashchange", synchronize);
      window.removeEventListener("popstate", synchronize);
    };
  }, []);

  useEffect(() => {
    if (section === "current") {
      setCurrentWorkspaceMounted(true);
    }
  }, [section]);

  const selectSection = (next: ProductSection) => {
    setSection(next);
    setNavExpanded(false);
    if (window.location.hash !== `#${next}`) {
      window.history.pushState(null, "", `#${next}`);
    }
  };

  const setCurrentJob = useCallback((jobId: string) => {
    currentJobIdRef.current = jobId;
    setCurrentJobId(jobId);
    if (jobId) {
      window.localStorage.setItem(CURRENT_DISCOVERY_STORAGE_KEY, jobId);
    } else {
      window.localStorage.removeItem(CURRENT_DISCOVERY_STORAGE_KEY);
    }
  }, []);

  const onSeedBatchInputs = useCallback(
    (
      handoff: Pick<
        DiscoveryBatchHandoff,
        "inputs" | "input_records" | "job_id" | "discovery_id" | "batch_index"
      >,
    ) => {
      const next = (handoff.inputs || []).join("\n").trim();
      if (!next) return;
      setBatchSeedInputs(next);
      setBatchSeedRecords(handoff.input_records || []);
      setBatchSeedSource({
        jobId: handoff.job_id,
        discoveryId: handoff.discovery_id,
        batchIndex: handoff.batch_index,
      });
      setBatchSeedToken((token) => token + 1);
      selectSection("batch");
    },
    [],
  );

  const chatVisible = !currentJobId || operationsTab === "chat";

  return (
    <>
      <Header aria-label="PRIDE 蛋白质组学数据运行平台">
        <SkipToContent />
        <HeaderMenuButton
          aria-label={navExpanded ? "关闭产品导航" : "打开产品导航"}
          isActive={navExpanded}
          onClick={() => setNavExpanded((value) => !value)}
          isCollapsible
        />
        <HeaderName
          href="#current"
          prefix="PRIDE"
          onClick={(event) => {
            event.preventDefault();
            selectSection("current");
          }}
        >
          蛋白质组学数据运行平台
        </HeaderName>
        <div className="product-build">
          <Tag type="cool-gray" title={buildInfoTitle()}>
            Build {buildInfoLabel()}
          </Tag>
        </div>
      </Header>
      <SideNav
        aria-label="产品导航"
        expanded={navExpanded}
        isPersistent
        onOverlayClick={() => setNavExpanded(false)}
      >
        <SideNavItems>
          {navItems.slice(0, 2).map((item) => (
            <SideNavLink
              key={item.id}
              href={`#${item.id}`}
              isActive={section === item.id}
              onClick={(event) => {
                event.preventDefault();
                selectSection(item.id);
              }}
            >
              {item.label}
            </SideNavLink>
          ))}
          <SideNavDivider />
          {navItems.slice(2).map((item) => (
            <SideNavLink
              key={item.id}
              href={`#${item.id}`}
              isActive={section === item.id}
              onClick={(event) => {
                event.preventDefault();
                selectSection(item.id);
              }}
            >
              {item.label}
            </SideNavLink>
          ))}
        </SideNavItems>
      </SideNav>

      <Content id="main-content" className="product-content" tabIndex={-1}>
        {currentWorkspaceMounted ? (
          <section
            className="current-task-page"
            aria-label="当前任务"
            hidden={section !== "current"}
          >
          {currentJobId ? (
            <div className="new-task-row">
              <Button
                kind="ghost"
                size="sm"
                onClick={() => {
                  setCurrentJob("");
                  setOperationsTab("overview");
                  setLegacyJob(null);
                  setPhase("idle");
                }}
              >
                新建发现任务
              </Button>
            </div>
          ) : (
            <header className="page-title page-title--discovery">
              <div>
                <p className="ops-eyebrow">DISCOVERY WORKBENCH</p>
                <h1>定义数据发现目标</h1>
                <p>
                  用自然语言描述您的检索目标，系统将自动生成检索词并执行发现任务。
                  确认检索词之前不会访问 PRIDE 数据库。
                </p>
              </div>
            </header>
          )}

          <div
            className={`current-task-workspace${
              currentJobId ? " current-task-workspace--running" : ""
            }`}
          >
              <div
                className="persistent-chat"
                aria-hidden={!chatVisible}
                hidden={!chatVisible}
              >
                <Tile className="agent-surface">
                  <Suspense
                    fallback={
                      <div className="route-loading">
                        <InlineLoading description="正在加载任务对话…" />
                      </div>
                    }
                  >
                    <CarbonAgentChat
                      onJob={(next) => {
                        setLegacyJob(next);
                        const jobId = String(next.job_id || "");
                        // Legacy progress snapshots may arrive many times for
                        // the same durable job. They must not reset the tab the
                        // operator is currently inspecting.
                        if (jobId && jobId !== currentJobIdRef.current) {
                          setCurrentJob(jobId);
                          setOperationsTab("overview");
                        }
                      }}
                      onIntentChange={setIntent}
                      onPhaseChange={setPhase}
                      externalSelectedSearchTerms={selectedSearchTerms}
                      onNavigate={(index) =>
                        selectSection(sectionFromLegacyIndex(index))
                      }
                      onOpenResults={(jobId) => {
                        setCurrentJob(jobId);
                        setOperationsTab("batches");
                        setResultOpenRequest({
                          jobId,
                          token: Date.now(),
                        });
                      }}
                      externalCommand={externalCommand}
                      onExternalCommandConsumed={() => setExternalCommand(null)}
                    />
                  </Suspense>
                </Tile>
              </div>

            {!currentJobId ? (
              <DiscoveryContextRail
                spec={intent}
                phase={phase}
                job={legacyJob}
                onConfirm={(queryTerms) =>
                  setExternalCommand({ type: "confirm", queryTerms })
                }
                onIntentChange={setIntent}
                selectedSearchTerms={selectedSearchTerms}
                onSelectedSearchTermsChange={setSelectedSearchTerms}
                onApplyDefaults={() =>
                  setExternalCommand({ type: "defaults" })
                }
                onNavigate={(index) =>
                  selectSection(sectionFromLegacyIndex(index))
                }
                onSeedBatchInputs={onSeedBatchInputs}
                openResultRequest={resultOpenRequest}
              />
            ) : (
              <OperationsConsole
                jobId={currentJobId}
                activeTab={operationsTab}
                onTabChange={setOperationsTab}
                onSeedBatchInputs={onSeedBatchInputs}
              />
            )}
          </div>
          </section>
        ) : null}

        <Suspense
          fallback={
            <div className="route-loading">
              <InlineLoading description="正在加载工作区…" />
            </div>
          }
        >
          {section === "history" ? (
            <OperationsHistory
              onOpenJob={(jobId) => {
                setCurrentJob(jobId);
                setOperationsTab("overview");
                selectSection("current");
              }}
              onOpenTask={(id) => {
                setTaskId(id);
                selectSection("single");
              }}
              onOpenBatch={(id) => {
                setBatchId(id);
                selectSection("batch");
              }}
            />
          ) : null}
          {section === "single" ? (
            <SingleTaskPanel
              taskId={taskId}
              onTaskId={setTaskId}
              onChanged={() => undefined}
            />
          ) : null}
          {section === "batch" ? (
            <BatchPanel
              batchId={batchId}
              onBatchId={setBatchId}
              onChanged={() => undefined}
              initialInputs={batchSeedInputs}
              initialInputsToken={batchSeedToken}
              initialInputRecords={batchSeedRecords}
              initialSource={batchSeedSource}
            />
          ) : null}
          {section === "ai-ready" ? (
            <AiReadyPanel onChanged={() => undefined} />
          ) : null}
          {section === "settings" ? <SettingsPanel /> : null}
        </Suspense>
      </Content>
    </>
  );
}
