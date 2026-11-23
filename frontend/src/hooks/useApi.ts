import { useCallback, useEffect, useRef, useState } from 'react'
import { errorMessage } from '../services/api'

export interface ApiState<T> {
  data: T | null
  error: string | null
  loading: boolean
  /** Re-run the request. `silent` keeps the current data visible (no loading state). */
  refetch: (silent?: boolean) => Promise<void>
  setData: (updater: T | null | ((prev: T | null) => T | null)) => void
}

/**
 * Run an API call whenever `deps` change; aborts in-flight requests on change/unmount.
 * Pass `enabled = false` to skip the request.
 */
export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: ReadonlyArray<unknown>,
  enabled = true,
): ApiState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(enabled)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher
  const controllerRef = useRef<AbortController | null>(null)

  const run = useCallback(async (silent = false) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    if (!silent) setLoading(true)
    try {
      const result = await fetcherRef.current(controller.signal)
      if (!controller.signal.aborted) {
        setData(result)
        setError(null)
      }
    } catch (err) {
      if (controller.signal.aborted) return
      setError(errorMessage(err))
    } finally {
      if (!controller.signal.aborted) setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!enabled) {
      setLoading(false)
      return
    }
    void run()
    return () => controllerRef.current?.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, run, ...deps])

  return { data, error, loading, refetch: run, setData }
}
