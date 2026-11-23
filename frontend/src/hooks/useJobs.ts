import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, errorMessage, getJob } from '../services/api'
import type { Job } from '../types/api'

export interface TrackedJob {
  id: string
  label: string
  job: Job | null
  error: string | null
}

const isDone = (j: Job | null) => j !== null && (j.status === 'completed' || j.status === 'failed')

/**
 * Track several background jobs (GET /api/jobs/{id}) until they complete or fail.
 * `onSettled` fires once per job when it reaches a terminal state.
 */
export function useJobTracker(onSettled?: (job: Job, label: string) => void, intervalMs = 1200) {
  const [jobs, setJobs] = useState<TrackedJob[]>([])
  const settledRef = useRef(new Set<string>())
  const onSettledRef = useRef(onSettled)
  onSettledRef.current = onSettled
  const jobsRef = useRef(jobs)
  jobsRef.current = jobs

  const track = useCallback((id: string, label: string) => {
    setJobs((prev) => (prev.some((j) => j.id === id) ? prev : [...prev, { id, label, job: null, error: null }]))
  }, [])

  const dismiss = useCallback((id: string) => {
    setJobs((prev) => prev.filter((j) => j.id !== id))
  }, [])

  const active = jobs.some((j) => !isDone(j.job) && !j.error)

  useEffect(() => {
    if (!active) return
    let cancelled = false
    const tick = async () => {
      const pending = jobsRef.current.filter((j) => !isDone(j.job) && !j.error)
      const updates = await Promise.all(
        pending.map(async (j) => {
          try {
            return { id: j.id, job: await getJob(j.id), error: null as string | null }
          } catch (err) {
            return { id: j.id, job: null, error: errorMessage(err) }
          }
        }),
      )
      if (cancelled) return
      setJobs((prev) =>
        prev.map((j) => {
          const u = updates.find((x) => x.id === j.id)
          if (!u) return j
          return { ...j, job: u.job ?? j.job, error: u.error }
        }),
      )
      for (const u of updates) {
        if (u.job && isDone(u.job) && !settledRef.current.has(u.id)) {
          settledRef.current.add(u.id)
          const label = jobsRef.current.find((j) => j.id === u.id)?.label ?? u.id
          onSettledRef.current?.(u.job, label)
        }
      }
    }
    void tick()
    const id = window.setInterval(() => void tick(), intervalMs)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [active, intervalMs])

  return { jobs, track, dismiss, active }
}

/** Poll a single job id until it settles. */
export function useJob(jobId: string | null, onSettled?: (job: Job) => void, intervalMs = 1500) {
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const cb = useRef(onSettled)
  cb.current = onSettled

  useEffect(() => {
    setJob(null)
    setError(null)
    if (!jobId) return
    let cancelled = false
    let timer: number | undefined
    const tick = async () => {
      try {
        const j = await getJob(jobId)
        if (cancelled) return
        setJob(j)
        setError(null)
        if (isDone(j)) {
          cb.current?.(j)
          return
        }
      } catch (err) {
        if (cancelled) return
        setError(errorMessage(err))
        // unknown job ids will never appear: stop polling
        if (err instanceof ApiError && err.status === 404) return
      }
      timer = window.setTimeout(() => void tick(), intervalMs)
    }
    void tick()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [jobId, intervalMs])

  return { job, error }
}
