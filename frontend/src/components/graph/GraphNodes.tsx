import { memo, type CSSProperties, type ReactNode } from 'react'
import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import { CircleHelp, FileText, Rows3, Tag } from 'lucide-react'
import type { GraphNode } from '../../types/api'
import { NODE_TYPE_COLORS, entityColor } from '../../lib/constants'
import { cx } from '../../lib/format'

export interface FlowNodeData extends Record<string, unknown> {
  node: GraphNode
  dimmed: boolean
  matched: boolean
  emphasized: boolean
}

export type FlowNode = Node<FlowNodeData>

/** Invisible centred handles so straight edges connect node centres (network-style layout). */
const handleStyle: CSSProperties = {
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 1,
  height: 1,
  minWidth: 0,
  minHeight: 0,
  border: 0,
  background: 'transparent',
  opacity: 0,
  pointerEvents: 'none',
}

function Handles() {
  return (
    <>
      <Handle type="target" position={Position.Top} style={handleStyle} isConnectable={false} />
      <Handle type="source" position={Position.Bottom} style={handleStyle} isConnectable={false} />
    </>
  )
}

function Shell({ data, selected, color, icon, children, className }: { data: FlowNodeData; selected: boolean; color: string; icon: ReactNode; children: ReactNode; className?: string }) {
  const ring = selected ? `0 0 0 2px ${color}` : data.emphasized ? `0 0 0 1.5px ${color}, 0 0 18px ${color}55` : data.matched ? '0 0 0 2px #fde047' : undefined
  return (
    <div
      className={cx('flex items-center gap-1.5 border bg-[#0e131d] text-fg transition-opacity duration-200', className)}
      style={{ borderColor: `${color}${data.emphasized || selected ? 'cc' : '66'}`, boxShadow: ring, opacity: data.dimmed ? 0.18 : 1 }}
      title={data.node.label}
    >
      <Handles />
      <span className="flex shrink-0 items-center justify-center" style={{ color }}>
        {icon}
      </span>
      {children}
    </div>
  )
}

function QuestionNodeImpl({ data, selected }: NodeProps<FlowNode>) {
  const color = NODE_TYPE_COLORS.question
  return (
    <Shell data={data} selected={selected} color={color} icon={<CircleHelp size={16} />} className="max-w-[260px] rounded-xl px-3 py-2">
      <span className="line-clamp-3 text-xs leading-snug font-semibold">{data.node.label}</span>
    </Shell>
  )
}

function DocumentNodeImpl({ data, selected }: NodeProps<FlowNode>) {
  const color = NODE_TYPE_COLORS.document
  return (
    <Shell data={data} selected={selected} color={color} icon={<FileText size={14} />} className="max-w-[200px] rounded-md px-2.5 py-1.5">
      <span className="truncate text-[11px] font-semibold">{data.node.label}</span>
    </Shell>
  )
}

function PassageNodeImpl({ data, selected }: NodeProps<FlowNode>) {
  const color = NODE_TYPE_COLORS.passage
  const section = typeof data.node.data.section === 'string' ? data.node.data.section : null
  return (
    <Shell data={data} selected={selected} color={color} icon={<Rows3 size={13} />} className="max-w-[190px] rounded-md px-2 py-1.5">
      <span className="min-w-0">
        <span className="block truncate text-[11px] font-medium">{data.node.label}</span>
        {section && <span className="block truncate text-[10px] text-muted">{section}</span>}
      </span>
    </Shell>
  )
}

function EntityNodeImpl({ data, selected }: NodeProps<FlowNode>) {
  const color = entityColor(data.node.data.category)
  const isQ = data.node.data.is_question_entity === true
  return (
    <Shell data={data} selected={selected} color={color} icon={<Tag size={11} />} className={cx('max-w-[170px] rounded-full px-2 py-0.5', isQ && 'border-dashed')}>
      <span className="truncate text-[11px]">{data.node.label}</span>
    </Shell>
  )
}

export const nodeTypes = {
  question: memo(QuestionNodeImpl),
  document: memo(DocumentNodeImpl),
  passage: memo(PassageNodeImpl),
  entity: memo(EntityNodeImpl),
}
