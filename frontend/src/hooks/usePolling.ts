import { useEffect, useRef } from 'react'

/** Call `callback` every `intervalMs` while `active` is true (not immediately). */
export function usePolling(callback: () => void, intervalMs: number, active = true): void {
  const ref = useRef(callback)
  ref.current = callback
  useEffect(() => {
    if (!active) return
    const id = window.setInterval(() => ref.current(), intervalMs)
    return () => window.clearInterval(id)
  }, [intervalMs, active])
}
