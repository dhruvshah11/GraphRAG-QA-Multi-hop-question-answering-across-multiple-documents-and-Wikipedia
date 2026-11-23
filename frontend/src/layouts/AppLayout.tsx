import { useState, type ReactNode } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  BarChart3,
  FileText,
  FlaskConical,
  LayoutDashboard,
  Menu,
  MessageSquareText,
  Network,
  Settings,
  X,
} from 'lucide-react'
import { HealthIndicator } from '../components/HealthIndicator'
import { cx } from '../lib/format'

const NAV: Array<{ to: string; label: string; icon: ReactNode; end?: boolean }> = [
  { to: '/', label: 'Dashboard', icon: <LayoutDashboard size={17} />, end: true },
  { to: '/documents', label: 'Documents', icon: <FileText size={17} /> },
  { to: '/qa', label: 'Question Answering', icon: <MessageSquareText size={17} /> },
  { to: '/graph', label: 'Graph Explorer', icon: <Network size={17} /> },
  { to: '/analytics', label: 'Retrieval & GNN', icon: <BarChart3 size={17} /> },
  { to: '/evaluation', label: 'Evaluation Lab', icon: <FlaskConical size={17} /> },
  { to: '/settings', label: 'Settings', icon: <Settings size={17} /> },
]

function Brand() {
  return (
    <div className="flex items-center gap-2.5 px-2">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/90">
        <svg viewBox="0 0 32 32" className="h-5 w-5" aria-hidden>
          <g stroke="#e0e7ff" strokeWidth="2.2" fill="none">
            <path d="M9 10h14M9 10l7 12M23 10l-7 12" />
          </g>
          <g fill="#fff">
            <circle cx="9" cy="10" r="3.2" />
            <circle cx="23" cy="10" r="3.2" />
            <circle cx="16" cy="22" r="3.2" />
          </g>
        </svg>
      </div>
      <div className="leading-tight">
        <div className="text-sm font-semibold tracking-tight text-fg">GraphRAG-QA</div>
        <div className="text-[11px] text-subtle">Multi-hop evidence graphs</div>
      </div>
    </div>
  )
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <div className="py-5">
        <Brand />
      </div>
      <nav className="flex-1 space-y-0.5" aria-label="Main">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            onClick={onNavigate}
            className={({ isActive }) =>
              cx(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive ? 'bg-accent-soft text-indigo-200' : 'text-muted hover:bg-surface-2 hover:text-fg',
              )
            }
          >
            {item.icon}
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-line py-4">
        <HealthIndicator />
      </div>
    </div>
  )
}

export function AppLayout() {
  const [open, setOpen] = useState(false)
  const location = useLocation()
  const fullBleed = location.pathname.startsWith('/graph')

  return (
    <div className="min-h-screen bg-bg">
      {/* desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 border-r border-line bg-surface/70 px-3 lg:block">
        <SidebarContent />
      </aside>

      {/* mobile top bar */}
      <div className="sticky top-0 z-30 flex items-center justify-between border-b border-line bg-surface/95 px-4 py-2.5 backdrop-blur lg:hidden">
        <Brand />
        <button type="button" className="btn-ghost btn-sm" onClick={() => setOpen(true)} aria-label="Open navigation">
          <Menu size={18} />
        </button>
      </div>

      {/* mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <aside className="relative h-full w-64 animate-fade-in border-r border-line bg-surface px-3">
            <button type="button" className="btn-ghost btn-sm absolute top-5 right-2" onClick={() => setOpen(false)} aria-label="Close navigation">
              <X size={16} />
            </button>
            <SidebarContent onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}

      <main className="lg:pl-60">
        <div className={cx('mx-auto px-4 py-6 sm:px-6 lg:px-8', fullBleed ? 'max-w-none' : 'max-w-[1400px]')}>
          <Outlet />
        </div>
      </main>
    </div>
  )
}
