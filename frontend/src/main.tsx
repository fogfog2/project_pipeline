import { useEffect, useMemo, useState, type FormEvent } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Project = { id: string; name: string; description: string; task_kind: string; mode?: string; recipe_id?: string; status?: string };
type Model = { id: string; name: string; version: string; family: string; alias?: string; precision: string; format: string; runnable: boolean; status?: string };
type Run = { id: string; kind: string; name: string; status: string; metrics: Record<string, number>; model_id?: string; dataset_id?: string; notes: string };
type Dataset = { id: string; name: string; version: string; format: string; status: string; sample_count: number; validation: Record<string, unknown> };
type Job = { id: string; runner_id: string; status: string; log: string; result_json: Record<string, unknown> };
type Target = { id: string; name: string; version: string; target_kind: string; runtime: string; hardware: Record<string, unknown>; notes: string };
type StorageMapping = { id: string; name: string; root_path: string; read_only: boolean; status: string; last_validation: Record<string, unknown>; notes: string };
type Asset = { id: string; relative_path: string; size_bytes: number; sha256?: string; status: string };
type Overview = { counts: Record<string, number>; lineage_completeness: number; baseline?: Model; candidate?: Model; recent_runs: Run[]; next_actions: string[] };

const isStatic = import.meta.env.VITE_STATIC_MODE === "true";
const api = async <T,>(path: string, options?: RequestInit): Promise<T> => {
  if (isStatic) {
    if (options?.method && options.method !== "GET") throw new Error("정적 Pages 모드에서는 조회만 가능합니다.");
    const snapshot = await fetch(`${import.meta.env.BASE_URL}snapshot.json`).then((response) => response.json());
    if (path === "/projects") return [snapshot.project] as T;
    if (path.endsWith("/overview")) return snapshot.overview as T;
    if (path.endsWith("/models")) return snapshot.models as T;
    if (path.endsWith("/runs")) return snapshot.runs as T;
    if (path.endsWith("/datasets")) return (snapshot.datasets || []) as T;
    if (path.endsWith("/jobs")) return (snapshot.jobs || []) as T;
    if (path.endsWith("/targets")) return (snapshot.targets || []) as T;
    if (path.endsWith("/storages")) return (snapshot.storages || []) as T;
    throw new Error("정적 snapshot에 없는 API입니다.");
  }
  const response = await fetch(`/api/v1${path}`, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
};

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<string>();
  const [overview, setOverview] = useState<Overview>();
  const [models, setModels] = useState<Model[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [targets, setTargets] = useState<Target[]>([]);
  const [storages, setStorages] = useState<StorageMapping[]>([]);
  const [page, setPage] = useState("개요");
  const [message, setMessage] = useState("RTMDet · YOLOX 온보딩 예제를 시작하거나 기존 자료를 연결하세요.");

  const selected = useMemo(() => projects.find((project) => project.id === projectId), [projects, projectId]);
  const loadProject = async (id: string) => {
    const [nextOverview, nextModels, nextRuns, nextDatasets, nextJobs, nextTargets, nextStorages] = await Promise.all([
      api<Overview>(`/projects/${id}/overview`), api<Model[]>(`/projects/${id}/models`), api<Run[]>(`/projects/${id}/runs`), api<Dataset[]>(`/projects/${id}/datasets`), api<Job[]>(`/projects/${id}/jobs`), api<Target[]>(`/projects/${id}/targets`), api<StorageMapping[]>(`/projects/${id}/storages`)
    ]);
    setProjectId(id); setOverview(nextOverview); setModels(nextModels); setRuns(nextRuns); setDatasets(nextDatasets); setJobs(nextJobs); setTargets(nextTargets); setStorages(nextStorages);
  };
  const loadProjects = async () => {
    const value = await api<Project[]>("/projects");
    setProjects(value);
    if (value.length && !projectId) await loadProject(value[0].id);
  };
  useEffect(() => { void loadProjects().catch((error) => setMessage(`API 연결 오류: ${error.message}`)); }, []);
  const createRecipeProject = async () => {
    try {
      const project = await api<Project>("/projects", { method: "POST", body: JSON.stringify({
        name: `MMDetection·YOLOX 실습 ${new Date().toISOString().slice(0, 16).replace("T", " ")}`,
        description: "가이드에 따라 빈 프로젝트에서 데이터·모델·평가를 연결하는 선택형 실습",
        task_kind: "detection", mode: "guided", recipe_id: "mmdetection-onboarding",
      }) });
      await loadProjects(); await loadProject(project.id);
      setMessage("빈 실습 프로젝트를 만들었습니다. 먼저 데이터 화면에서 이미지·annotation 경로를 연결하세요. 모델과 점수는 아직 등록되지 않았습니다.");
    } catch (error) { setMessage(`실습 프로젝트 생성 오류: ${String(error)}`); }
  };
  const createBlankProject = async (name: string, task: string) => {
    try {
      const project = await api<Project>("/projects", { method: "POST", body: JSON.stringify({ name, task_kind: task, mode: "user", recipe_id: "blank" }) });
      await loadProjects(); await loadProject(project.id);
      setMessage("빈 프로젝트를 만들었습니다. 다음 단계부터 자료를 직접 연결하세요.");
    } catch (error) { setMessage(`프로젝트 생성 오류: ${String(error)}`); }
  };
  const compare = async () => {
    if (!overview?.baseline?.id || !overview?.candidate?.id) return;
    const value = await api<{ compatible: boolean; delta: Record<string, number>; reason?: string }>("/comparisons", {
      method: "POST", body: JSON.stringify({ baseline_model_id: overview.baseline.id, candidate_model_id: overview.candidate.id })
    });
    setMessage(value.compatible ? `비교 완료: ${Object.entries(value.delta).map(([key, delta]) => `${key} ${delta > 0 ? "+" : ""}${delta}`).join(", ")}` : value.reason || "비교할 수 없습니다.");
    setPage("평가·비교");
  };
  const runMockBoard = async () => {
    if (!projectId) return;
    await api(`/projects/${projectId}/jobs`, { method: "POST", body: JSON.stringify({ runner_id: "mock-board", input_json: { source: "dashboard" } }) });
    await loadProject(projectId);
    setMessage("모의 보드 작업을 생성했습니다. 이 결과는 하드웨어 측정값이 아닙니다.");
  };
  const archive = async (kind: "datasets" | "models", id: string) => {
    if (!projectId || !window.confirm("원본 파일과 과거 결과는 삭제하지 않고 이 항목을 보관 처리합니다. 계속할까요?")) return;
    try { await api(`/projects/${projectId}/${kind}/${id}/archive`, { method: "POST" }); await loadProject(projectId); setMessage("보관 상태를 변경했습니다. 필요하면 같은 메뉴에서 복원할 수 있습니다."); } catch (error) { setMessage(`보관 상태 변경 오류: ${String(error)}`); }
  };
  const finalizeDataset = async (id: string) => {
    if (!projectId || !window.confirm("현재 annotation/manifest 내용의 fingerprint를 고정합니다. 이후 수정은 새 DatasetVersion으로 등록해야 합니다. 계속할까요?")) return;
    try {
      await api(`/projects/${projectId}/datasets/${id}/finalize`, { method: "POST" });
      await loadProject(projectId);
      setMessage("DatasetVersion을 확정했습니다. 이후 원본 내용이 달라지면 기존 평가를 실행할 수 없습니다.");
    } catch (error) { setMessage(`Dataset 확정 오류: ${String(error)}`); }
  };

  const nav = ["개요", "데이터", "실험", "모델", "평가·비교", "실행·보드", "Release·리포트", "연결·설정", "가이드"];
  return <div className="shell">
    <aside><div className="brand">VISION<br/><b>LIFECYCLE</b></div><button className="demo" disabled={isStatic} onClick={() => void createRecipeProject()}>실습 프로젝트 시작</button>
      <nav>{nav.map((item) => <button className={page === item ? "active" : ""} onClick={() => setPage(item)} key={item}>{item}</button>)}</nav>
      <small>Local mode · API v1</small></aside>
    <main><header><div><p className="eyebrow">PROJECT / {selected?.task_kind || "SETUP"}</p><h1>{selected?.name || "Vision AI Lifecycle"}</h1>{selected && <small className="muted">{selected.mode === "guided" ? "Guided practice · 자료와 결과를 단계별로 연결" : "User project · 직접 연결"}</small>}</div><div className="header-actions"><button className="secondary" disabled={isStatic} onClick={() => setProjectId(undefined)}>새 프로젝트</button><select aria-label="프로젝트 선택" value={projectId || ""} onChange={(event) => void loadProject(event.target.value)}><option value="">프로젝트 선택</option>{projects.map((project) => <option value={project.id} key={project.id}>{project.name}</option>)}</select></div></header>
      <p className="notice">{message}</p>
      {!projectId ? <ProjectStarter onCreated={(name, task) => void createBlankProject(name, task)} onRecipe={() => void createRecipeProject()} onError={setMessage}/> : <>
      {page === "개요" && <><section className="cards"><Metric label="Dataset versions" value={overview?.counts.datasets ?? 0}/><Metric label="Model versions" value={overview?.counts.models ?? 0}/><Metric label="Evaluation runs" value={overview?.counts.runs ?? 0}/><Metric label="Lineage completeness" value={`${overview?.lineage_completeness ?? 0}%`}/></section>
      <section className="grid2"><article><h2>Baseline · Candidate</h2><div className="comparison"><div><label>BASELINE</label><strong>{overview?.baseline?.family || "미지정"}</strong><span>{overview?.baseline?.version}</span></div><div><label>CANDIDATE</label><strong>{overview?.candidate?.family || "미지정"}</strong><span>{overview?.candidate?.version}</span></div></div><button disabled={!overview?.baseline || !overview?.candidate} onClick={() => void compare()}>공통 평가 결과 비교</button></article><article><h2>다음 작업</h2><ol>{overview?.next_actions.map((item) => <li key={item}>{item}</li>)}</ol></article></section>
      <section><h2>최근 실행</h2><RunTable runs={runs.slice(0, 5)} /></section></>}
      {page === "모델" && <><section><h2>모델 Registry</h2><p>가중치, config, 전후처리, class mapping을 하나의 model bundle로 관리합니다. 학습 데이터는 자동 연결하지 않습니다.</p><ModelTable models={models} onArchive={(id) => void archive("models", id)}/></section><ModelConnect projectId={projectId} datasets={datasets} onSaved={() => void loadProject(projectId)} onError={setMessage}/></>}
      {page === "실험" && <section><h2>외부 Run 및 평가 실행</h2><RunTable runs={runs}/></section>}
      {page === "평가·비교" && <><section><h2>동일 평가 규약 비교</h2><p>공식 delta는 dataset version, evaluator version, class mapping version이 일치할 때만 계산합니다.</p><ModelTable models={models} onArchive={(id) => void archive("models", id)}/><button disabled={!overview?.baseline || !overview?.candidate} onClick={() => void compare()}>Baseline과 Candidate 비교</button></section><EvaluationConnect projectId={projectId} models={models} datasets={datasets} onSaved={() => void loadProject(projectId)} onError={setMessage}/></>}
      {page === "데이터" && <><section><h2>Dataset Version</h2><p>먼저 서버/NAS 경로를 검사한 다음 검사 결과를 초안 DatasetVersion으로 저장합니다. 확정 버전은 수정하지 않고 새 버전을 만듭니다.</p><DatasetTable datasets={datasets} onArchive={(id) => void archive("datasets", id)} onFinalize={(id) => void finalizeDataset(id)}/><DatasetConnect projectId={projectId} onSaved={() => void loadProject(projectId)} onError={setMessage}/></section><PathInspector projectId={projectId} onError={setMessage}/></>}
      {page === "가이드" && <Guide />}
      {page === "실행·보드" && <><section><h2>실행·보드</h2><p>보드와 runtime은 버전이 있는 Target Profile로 기록합니다. 실제 benchmark manifest에는 해당 profile ID를 연결합니다.</p><TargetTable targets={targets}/>{!isStatic && <TargetConnect projectId={projectId} onSaved={() => void loadProject(projectId)} onError={setMessage}/>}</section><section><h2>등록 작업</h2><p>등록 runner만 실행할 수 있습니다. 모의 board runner는 result contract 검증용입니다.</p><button disabled={isStatic} onClick={() => void runMockBoard()}>모의 보드 실행</button><JobTable jobs={jobs}/></section></>}
      {page === "Release·리포트" && <section><h2>Release·리포트</h2><p>Gate는 필수 지표가 없으면 INCOMPLETE로 처리합니다. JSON export API는 모델과 데이터의 절대 경로를 제거합니다.</p><code>GET /api/v1/projects/{projectId}/export</code></section>}
      {page === "연결·설정" && <><StorageConnect projectId={projectId} storages={storages} onSaved={() => void loadProject(projectId)} onError={setMessage}/><AgentPrompt projectId={projectId} onError={setMessage}/></>}
      </>}</main></div>;
}

function ModelTable({ models, onArchive }: { models: Model[]; onArchive?: (id: string) => void }) { return <div className="tablewrap"><table><thead><tr><th>Family</th><th>Version</th><th>Format</th><th>Precision</th><th>Alias</th><th>Status</th><th>Action</th></tr></thead><tbody>{models.map((model) => <tr key={model.id}><td><b>{model.family}</b></td><td>{model.version}</td><td>{model.format}</td><td>{model.precision}</td><td>{model.alias || "—"}</td><td><span className="pill">{model.status || (model.runnable ? "runnable" : "setup-required")}</span></td><td>{onArchive && <button className="table-action" onClick={() => onArchive(model.id)}>{model.status === "archived" ? "복원" : "보관"}</button>}</td></tr>)}</tbody></table></div>; }
function RunTable({ runs }: { runs: Run[] }) { return <div className="tablewrap"><table><thead><tr><th>Run</th><th>Kind</th><th>Status</th><th>bbox mAP</th><th>P50 latency</th></tr></thead><tbody>{runs.map((run) => <tr key={run.id}><td>{run.name}</td><td>{run.kind}</td><td><span className="pill">{run.status}</span></td><td>{run.metrics.bbox_mAP?.toFixed(3) ?? "—"}</td><td>{run.metrics.latency_ms_p50 ? `${run.metrics.latency_ms_p50} ms` : "—"}</td></tr>)}</tbody></table></div>; }
function DatasetTable({ datasets, onArchive, onFinalize }: { datasets: Dataset[]; onArchive?: (id: string) => void; onFinalize?: (id: string) => void }) { return <div className="tablewrap"><table><thead><tr><th>Name</th><th>Version</th><th>Format</th><th>Samples</th><th>Status</th><th>Action</th></tr></thead><tbody>{datasets.length ? datasets.map((item) => <tr key={item.id}><td><b>{item.name}</b></td><td>{item.version}</td><td>{item.format}</td><td>{item.sample_count}</td><td><span className="pill">{item.status}</span></td><td>{onFinalize && item.status === "draft" && <button className="table-action" onClick={() => onFinalize(item.id)}>확정</button>}{onArchive && <button className="table-action" onClick={() => onArchive(item.id)}>{item.status === "archived" ? "복원" : "보관"}</button>}</td></tr>) : <tr><td colSpan={6}>등록된 DatasetVersion이 없습니다. annotation 또는 manifest 경로를 검사해 초안을 만드세요.</td></tr>}</tbody></table></div>; }
function JobTable({ jobs }: { jobs: Job[] }) { return <div className="tablewrap jobs"><table><thead><tr><th>Runner</th><th>Status</th><th>Log</th></tr></thead><tbody>{jobs.map((job) => <tr key={job.id}><td>{job.runner_id}</td><td><span className="pill">{job.status}</span></td><td>{job.log || "대기 중"}</td></tr>)}</tbody></table></div>; }
function TargetTable({ targets }: { targets: Target[] }) { return <div className="tablewrap"><table><thead><tr><th>Target</th><th>Version</th><th>Runtime</th><th>Hardware</th></tr></thead><tbody>{targets.length ? targets.map((target) => <tr key={target.id}><td><b>{target.name}</b><br/><small>{target.target_kind}</small></td><td>{target.version}</td><td>{target.runtime}</td><td>{Object.entries(target.hardware).map(([key, value]) => `${key}: ${value}`).join(", ") || "—"}</td></tr>) : <tr><td colSpan={4}>연결된 target profile이 없습니다.</td></tr>}</tbody></table></div>; }
function ProjectStarter({ onCreated, onRecipe, onError }: { onCreated: (name: string, task: string) => void; onRecipe: () => void; onError: (message: string) => void }) {
  const [name, setName] = useState(""); const [task, setTask] = useState("detection");
  const create = (event: FormEvent) => { event.preventDefault(); if (!name.trim()) { onError("프로젝트 이름을 입력하세요."); return; } onCreated(name.trim(), task); };
  return <section className="empty"><span className="step-label">START HERE · 0 / PROJECT</span><h2>빈 프로젝트에서 시작하세요</h2><p>먼저 프로젝트만 만든 뒤 가이드에 따라 데이터, 모델, 평가 결과를 직접 연결합니다. 실습도 처음에는 빈 화면으로 시작합니다.</p><form className="form" onSubmit={(event) => void create(event)}><input required value={name} onChange={(event) => setName(event.target.value)} placeholder="프로젝트 이름"/><select value={task} onChange={(event) => setTask(event.target.value)}><option value="detection">Detection</option><option value="classification">Classification</option><option value="unknown">아직 모름</option></select><button type="submit">빈 프로젝트 만들기</button></form><button className="secondary" disabled={isStatic} onClick={onRecipe}>MMDetection · YOLOX 실습 시작(빈 상태)</button></section>;
}
function DatasetConnect({ projectId, onSaved, onError }: { projectId: string; onSaved: () => void; onError: (message: string) => void }) {
  const [name, setName] = useState(""); const [version, setVersion] = useState("v1"); const [format, setFormat] = useState("coco"); const [annotation, setAnnotation] = useState(""); const [manifest, setManifest] = useState("");
  const save = async (event: FormEvent) => { event.preventDefault(); try { await api(`/projects/${projectId}/datasets`, { method: "POST", body: JSON.stringify({ name, version, format, task_kind: format === "classification" ? "classification" : "detection", annotation_path: annotation || null, manifest_path: manifest || null, status: "draft" }) }); setName(""); setAnnotation(""); setManifest(""); onSaved(); } catch (error) { onError(`Dataset 초안 저장 오류: ${String(error)}`); } };
  return <div className="subform"><h3>검사한 자료를 Dataset 초안으로 연결</h3><form className="form two" onSubmit={(event) => void save(event)}><input required value={name} onChange={(event) => setName(event.target.value)} placeholder="Dataset 이름"/><input required value={version} onChange={(event) => setVersion(event.target.value)} placeholder="v1"/><select value={format} onChange={(event) => setFormat(event.target.value)}><option value="coco">COCO JSON</option><option value="yolo">YOLO TXT</option><option value="classification">Classification CSV / folder</option></select><input value={annotation} onChange={(event) => setAnnotation(event.target.value)} placeholder="annotation 경로(선택)"/><input value={manifest} onChange={(event) => setManifest(event.target.value)} placeholder="image/manifest 경로(선택)"/><button type="submit">초안 저장</button></form></div>;
}
function PathInspector({ projectId, onError }: { projectId: string; onError: (message: string) => void }) {
  const [path, setPath] = useState(""); const [result, setResult] = useState<string>();
  const inspect = async (event: FormEvent) => { event.preventDefault(); try { const value = await api<{ detected_format: string; result: unknown }>(`/projects/${projectId}/inspect-path`, { method: "POST", body: JSON.stringify({ path }) }); setResult(JSON.stringify(value, null, 2)); } catch (error) { onError(`경로 검사 오류: ${String(error)}`); } };
  return <section><h2>경로 검사</h2><p>서버가 접근할 수 있는 로컬/NAS 경로를 검사합니다. 반복해서 사용할 위치는 먼저 연결·설정에서 Storage mapping으로 등록하세요. 이 검사는 저장·업로드·외부 명령을 실행하지 않습니다.</p><form className="form" onSubmit={(event) => void inspect(event)}><input required value={path} onChange={(event) => setPath(event.target.value)} placeholder="/data/project/annotations.json 또는 dataset directory"/><button type="submit">형식 검사</button></form>{result && <pre>{result}</pre>}</section>;
}
function ModelConnect({ projectId, datasets, onSaved, onError }: { projectId: string; datasets: Dataset[]; onSaved: () => void; onError: (message: string) => void }) {
  const [name, setName] = useState(""); const [version, setVersion] = useState("v1"); const [family, setFamily] = useState(""); const [config, setConfig] = useState(""); const [artifact, setArtifact] = useState(""); const [datasetId, setDatasetId] = useState("");
  const save = async (event: FormEvent) => { event.preventDefault(); try { await api(`/projects/${projectId}/models`, { method: "POST", body: JSON.stringify({ name, version, family, format: "external", config_path: config || null, artifact_path: artifact || null, source_dataset_id: datasetId || null, metadata_json: { provenance: config && artifact && datasetId ? "complete" : "partial_or_unknown" } }) }); setName(""); setFamily(""); onSaved(); } catch (error) { onError(`모델 등록 오류: ${String(error)}`); } };
  return <section><h2>외부 모델 연결</h2><p>framework를 추정하지 않습니다. 사용자가 모델 family, config, artifact, 학습 데이터 관계를 직접 선택합니다.</p><form className="form two" onSubmit={(event) => void save(event)}><input required value={name} onChange={(event) => setName(event.target.value)} placeholder="Model 이름"/><input value={version} onChange={(event) => setVersion(event.target.value)} placeholder="Version"/><input required value={family} onChange={(event) => setFamily(event.target.value)} placeholder="예: RTMDet-tiny, YOLOX-s, ResNet"/><select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">학습 데이터 unknown / 아직 연결 안 함</option>{datasets.map((dataset) => <option value={dataset.id} key={dataset.id}>{dataset.name} · {dataset.version}</option>)}</select><input value={config} onChange={(event) => setConfig(event.target.value)} placeholder="config 경로(선택)"/><input value={artifact} onChange={(event) => setArtifact(event.target.value)} placeholder="모델 artifact 경로(선택)"/><button type="submit">모델 초안 연결</button></form></section>;
}
function TargetConnect({ projectId, onSaved, onError }: { projectId: string; onSaved: () => void; onError: (message: string) => void }) {
  const [name, setName] = useState(""); const [runtime, setRuntime] = useState("TensorRT"); const [version, setVersion] = useState("v1"); const [hardware, setHardware] = useState("");
  const save = async (event: FormEvent) => { event.preventDefault(); try { await api(`/projects/${projectId}/targets`, { method: "POST", body: JSON.stringify({ name, version, runtime, hardware: hardware ? { description: hardware } : {} }) }); setName(""); setHardware(""); onSaved(); } catch (error) { onError(`Target profile 등록 오류: ${String(error)}`); } };
  return <form className="form two" onSubmit={(event) => void save(event)}><input required value={name} onChange={(event) => setName(event.target.value)} placeholder="예: Jetson Orin NX"/><input value={version} onChange={(event) => setVersion(event.target.value)} placeholder="Runtime / SDK version"/><input value={runtime} onChange={(event) => setRuntime(event.target.value)} placeholder="예: TensorRT"/><input value={hardware} onChange={(event) => setHardware(event.target.value)} placeholder="예: 15W, JetPack 6"/><button type="submit">Target profile 등록</button></form>;
}
function EvaluationConnect({ projectId, models, datasets, onSaved, onError }: { projectId: string; models: Model[]; datasets: Dataset[]; onSaved: () => void; onError: (message: string) => void }) {
  const [modelId, setModelId] = useState(""); const [datasetId, setDatasetId] = useState(""); const [predictionsPath, setPredictionsPath] = useState(""); const [protocol, setProtocol] = useState("onboarding_ap50");
  const save = async (event: FormEvent) => { event.preventDefault(); try { const result = await api<{ result: Record<string, unknown> }>(`/projects/${projectId}/evaluations/predictions`, { method: "POST", body: JSON.stringify({ model_id: modelId, dataset_id: datasetId, predictions_path: predictionsPath, protocol }) }); onSaved(); onError(`평가 완료: AP50 ${String(result.result.bbox_AP50 ?? "—")} · 원본 prediction을 동일 조건으로 다시 비교할 수 있습니다.`); } catch (error) { onError(`평가 실행 오류: ${String(error)}`); } };
  return <section><span className="step-label">STEP 5 · EVALUATION</span><h2>외부 prediction 평가</h2><p>학습 시스템이나 보드에서 만든 COCO prediction JSON을 등록합니다. 이 시스템은 학습을 실행하지 않으며, 선택한 DatasetVersion의 fingerprint와 평가 조건을 기록합니다.</p><form className="form two" onSubmit={(event) => void save(event)}><select required value={modelId} onChange={(event) => setModelId(event.target.value)}><option value="">평가 모델 선택</option>{models.filter((model) => model.status !== "archived").map((model) => <option value={model.id} key={model.id}>{model.family} · {model.version}</option>)}</select><select required value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">평가 Dataset 선택</option>{datasets.filter((dataset) => dataset.status !== "archived").map((dataset) => <option value={dataset.id} key={dataset.id}>{dataset.name} · {dataset.version}</option>)}</select><select value={protocol} onChange={(event) => setProtocol(event.target.value)}><option value="onboarding_ap50">빠른 AP50</option><option value="coco_full">공식 COCO AP@[.50:.95]</option></select><input required value={predictionsPath} onChange={(event) => setPredictionsPath(event.target.value)} placeholder="서버 prediction JSON 경로"/><button disabled={isStatic} type="submit">평가 실행</button></form></section>;
}
function StorageConnect({ projectId, storages, onSaved, onError }: { projectId: string; storages: StorageMapping[]; onSaved: () => void; onError: (message: string) => void }) {
  const [name, setName] = useState("dataset-root"); const [rootPath, setRootPath] = useState(""); const [notes, setNotes] = useState("");
  const [browser, setBrowser] = useState<{ storageId: string; relativePath: string; entries: Array<{ name: string; relative_path: string; kind: string; size?: number }> }>();
  const [inventory, setInventory] = useState<Asset[]>([]);
  const save = async (event: FormEvent) => { event.preventDefault(); try { await api(`/projects/${projectId}/storages`, { method: "POST", body: JSON.stringify({ name, root_path: rootPath, read_only: true, notes }) }); setRootPath(""); setNotes(""); onSaved(); } catch (error) { onError(`Storage mapping 등록 오류: ${String(error)}`); } };
  const validate = async (id: string) => { try { await api(`/projects/${projectId}/storages/${id}/validate`, { method: "POST" }); onSaved(); } catch (error) { onError(`Storage 검사 오류: ${String(error)}`); } };
  const browse = async (storageId: string, relativePath = "") => { try { const value = await api<{ relative_path: string; entries: Array<{ name: string; relative_path: string; kind: string; size?: number }> }>(`/projects/${projectId}/storages/${storageId}/browse`, { method: "POST", body: JSON.stringify({ relative_path: relativePath }) }); setBrowser({ storageId, relativePath: value.relative_path, entries: value.entries }); } catch (error) { onError(`Storage 탐색 오류: ${String(error)}`); } };
  const runInventory = async (storageId: string) => { try { const value = await api<{ assets: Asset[] }>(`/projects/${projectId}/storages/${storageId}/inventory`, { method: "POST", body: JSON.stringify({ recursive: true, limit: 1000 }) }); setInventory(value.assets); onSaved(); } catch (error) { onError(`파일 inventory 오류: ${String(error)}`); } };
  return <><section><span className="step-label">STEP 1 · STORAGE</span><h2>프로젝트 자료 위치 연결</h2><p>이미지, annotation, 모델, config가 있는 서버/NAS 디렉터리를 읽기 전용 mapping으로 등록합니다. 경로는 로컬 DB에만 저장되며 Pages export에서는 제외됩니다.</p><form className="form two" onSubmit={(event) => void save(event)}><input required value={name} onChange={(event) => setName(event.target.value)} placeholder="예: dataset-root"/><input required value={rootPath} onChange={(event) => setRootPath(event.target.value)} placeholder="서버/NAS 절대 경로, 예: /mnt/vision-data"/><input value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="용도(선택)"/><button disabled={isStatic} type="submit">연결 검사 후 등록</button></form></section>
  <section><h2>등록된 Storage mapping</h2><div className="tablewrap"><table><thead><tr><th>이름</th><th>상태</th><th>접근</th><th>메모</th><th>작업</th></tr></thead><tbody>{storages.length ? storages.map((storage) => <tr key={storage.id}><td><b>{storage.name}</b></td><td><span className="pill">{storage.status}</span></td><td>{storage.read_only ? "읽기 전용" : "읽기/쓰기"}</td><td>{storage.notes || "—"}</td><td><button className="table-action" disabled={isStatic} onClick={() => void validate(storage.id)}>다시 검사</button><button className="table-action" disabled={isStatic} onClick={() => void browse(storage.id)}>탐색</button><button className="table-action" disabled={isStatic} onClick={() => void runInventory(storage.id)}>파일 목록·hash</button></td></tr>) : <tr><td colSpan={5}>아직 연결된 디렉터리가 없습니다.</td></tr>}</tbody></table></div>{browser && <div className="subform"><h3>탐색: {storages.find((storage) => storage.id === browser.storageId)?.name} / {browser.relativePath || "."}</h3><div className="file-list">{browser.relativePath && <button className="table-action" onClick={() => void browse(browser.storageId, browser.relativePath.split("/").slice(0, -1).join("/"))}>상위 폴더</button>}{browser.entries.map((entry) => entry.kind === "directory" ? <button className="table-action" key={entry.relative_path} onClick={() => void browse(browser.storageId, entry.relative_path)}>📁 {entry.name}</button> : <span key={entry.relative_path}>📄 {entry.name}{entry.size !== undefined ? ` (${entry.size} B)` : ""}</span>)}</div></div>}{inventory.length > 0 && <div className="subform"><h3>검색된 파일 {inventory.length}개</h3><div className="tablewrap"><table><thead><tr><th>상대 경로</th><th>크기</th><th>SHA-256</th></tr></thead><tbody>{inventory.slice(0, 30).map((asset) => <tr key={asset.id}><td>{asset.relative_path}</td><td>{asset.size_bytes} B</td><td><code>{asset.sha256?.slice(0, 16)}…</code></td></tr>)}</tbody></table></div></div>}</section></>;
}
function AgentPrompt({ projectId, onError }: { projectId: string; onError: (message: string) => void }) {
  const [prompt, setPrompt] = useState("");
  const getPrompt = async () => { try { const result = await api<{ prompt: string }>(`/projects/${projectId}/agent-request`); setPrompt(result.prompt); } catch (error) { onError(`Agent 요청문 생성 오류: ${String(error)}`); } };
  return <section><h2>Agent 연결 요청문</h2><p>등록한 경로와 lineage 누락 항목을 사용해 자료 연결을 요청합니다. 자격증명은 저장하거나 표시하지 않습니다.</p><button onClick={() => void getPrompt()}>요청문 만들기</button>{prompt && <><pre>{prompt}</pre><button className="secondary" onClick={() => void navigator.clipboard.writeText(prompt)}>클립보드로 복사</button></>}</section>;
}
function Guide() { return <section><h2>RTMDet · YOLOX 온보딩</h2><ol className="guide"><li><b>환경 검사</b><span>MMDetection 3.3.0, MMDeploy, ONNX Runtime 설치 상태를 확인합니다.</span></li><li><b>공통 데이터 연결</b><span>COCO annotation과 이미지 경로, category mapping을 검증합니다.</span></li><li><b>모델 연결</b><span>RTMDet-tiny 또는 YOLOX-s config와 checkpoint를 model bundle로 등록합니다.</span></li><li><b>평가와 비교</b><span>같은 evaluation set에서 native·ONNX·board 결과를 분리해 비교합니다.</span></li></ol></section>; }

createRoot(document.getElementById("root")!).render(<App />);
