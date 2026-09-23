import { useEffect, useId, useState } from "react";
import {
  Button, ComposedModal, InlineLoading, InlineNotification, ModalBody,
  ModalFooter, ModalHeader, Select, SelectItem, TextInput,
} from "@carbon/react";
import { Folder, FolderOpen } from "@carbon/icons-react";
import { listDatasetDirectories, type DatasetDirectories } from "./workflow-api";

type Props = {
  repository: string;
  localDir: string;
  disabled?: boolean;
  onChange: (repository: string, localDir: string) => void;
};

export function DatasetSourcePicker({ repository, localDir, disabled, onChange }: Props) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [path, setPath] = useState("");
  const [listing, setListing] = useState<DatasetDirectories | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    setListing(null);
    listDatasetDirectories(path, controller.signal)
      .then((result) => { if (!controller.signal.aborted) setListing(result); })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "无法读取目录");
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [open, path]);

  const browse = () => {
    setPath(localDir);
    setListing(null);
    setError("");
    setOpen(true);
  };

  return (
    <div className="dataset-source-picker">
      <Select
        id={`${id}-source`}
        labelText="数据来源"
        size="sm"
        value={repository}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value, localDir)}
      >
        <SelectItem value="pride" text="从 PRIDE 搜索数据集" />
        <SelectItem value="local" text="使用本地数据集" />
        {!["pride", "local"].includes(repository) ? <SelectItem value={repository} text={repository} /> : null}
      </Select>
      {repository === "local" ? (
        <>
          <TextInput
            id={`${id}-directory`}
            labelText="数据集目录"
            size="sm"
            value={localDir}
            readOnly
            disabled={disabled}
            placeholder="点击选择数据集文件夹"
            helperText="选择后直接使用目录中的数据，跳过数据集查找和下载。"
            onClick={browse}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") { event.preventDefault(); browse(); }
            }}
          />
          <Button kind="tertiary" size="sm" renderIcon={FolderOpen} disabled={disabled} onClick={browse}>
            选择文件夹
          </Button>
        </>
      ) : null}
      <ComposedModal open={open} onClose={() => setOpen(false)} size="sm">
        <ModalHeader title="选择数据集文件夹" closeModal={() => setOpen(false)} />
        <ModalBody>
          <p>浏览运行服务可访问的文件夹。</p>
          <TextInput
            id={`${id}-browse-path`}
            labelText="目录路径"
            defaultValue={listing?.path || path}
            key={listing?.path || path}
            placeholder="输入路径后按 Enter 打开"
            onKeyDown={(event) => {
              if (event.key === "Enter") { event.preventDefault(); setPath(event.currentTarget.value.trim()); }
            }}
          />
          <div className="dataset-directory-actions">
            <Button kind="ghost" size="sm" disabled={loading || !listing?.parent} onClick={() => setPath(listing?.parent || "")}>
              上一级
            </Button>
            <Button kind="ghost" size="sm" disabled={loading} onClick={() => setPath("")}>默认目录</Button>
          </div>
          {loading ? <InlineLoading description="正在读取文件夹…" /> : null}
          {error ? <InlineNotification kind="error" title="无法打开文件夹" subtitle={error} hideCloseButton /> : null}
          {listing && !loading ? (
            <ul className="dataset-directory-list" aria-label="子文件夹">
              {listing.directories.map((directory) => (
                <li key={directory.path}>
                  <Button kind="ghost" size="sm" renderIcon={Folder} onClick={() => setPath(directory.path)}>
                    {directory.name}
                  </Button>
                </li>
              ))}
              {!listing.directories.length ? <li>没有子文件夹，可以选择当前目录。</li> : null}
            </ul>
          ) : null}
        </ModalBody>
        <ModalFooter>
          <Button kind="secondary" onClick={() => setOpen(false)}>取消</Button>
          <Button
            kind="primary"
            disabled={loading || !listing || Boolean(error)}
            onClick={() => {
              if (!listing || loading) return;
              onChange("local", listing.path);
              setOpen(false);
            }}
          >
            选择此文件夹
          </Button>
        </ModalFooter>
      </ComposedModal>
    </div>
  );
}
