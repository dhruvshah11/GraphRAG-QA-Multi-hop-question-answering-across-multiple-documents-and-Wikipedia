import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router-dom'
import { AppLayout } from './layouts/AppLayout'
import { SkeletonBlock } from './components/Feedback'
import { EmptyState } from './components/Feedback'
import { Link } from 'react-router-dom'

const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const DocumentsPage = lazy(() => import('./pages/DocumentsPage'))
const QAPage = lazy(() => import('./pages/QAPage'))
const GraphPage = lazy(() => import('./pages/GraphPage'))
const AnalyticsPage = lazy(() => import('./pages/AnalyticsPage'))
const EvaluationPage = lazy(() => import('./pages/EvaluationPage'))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))

function PageFallback() {
  return (
    <div className="space-y-4 pt-2">
      <div className="skeleton h-7 w-56" />
      <SkeletonBlock rows={6} />
    </div>
  )
}

function NotFound() {
  return (
    <EmptyState
      title="Page not found"
      description="The page you are looking for does not exist."
      action={
        <Link to="/" className="btn-secondary">
          Back to dashboard
        </Link>
      }
    />
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        {(
          [
            ['/', DashboardPage],
            ['/documents', DocumentsPage],
            ['/qa', QAPage],
            ['/graph', GraphPage],
            ['/analytics', AnalyticsPage],
            ['/evaluation', EvaluationPage],
            ['/settings', SettingsPage],
          ] as const
        ).map(([path, Page]) => (
          <Route
            key={path}
            path={path}
            element={
              <Suspense fallback={<PageFallback />}>
                <Page />
              </Suspense>
            }
          />
        ))}
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
