import Foundation
import ScreenCaptureKit

/// Shared SCStream delegate: logs stream errors.
final class StreamDelegate: NSObject, SCStreamDelegate {
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("BsafeCapture: stream stopped with error: \(error)\n", stderr)
        exit(3)
    }
}

/// Synchronously resolve the displays selected by the --display filter.
/// Same selection semantics and errors as the original implementation.
/// Returns the selected displays plus the full shareable content (for app exclusion).
func resolveDisplays(filter rawFilter: String?) throws -> (displays: [SCDisplay], content: SCShareableContent) {
    var fetched: SCShareableContent?
    var fetchError: Error?
    let sem = DispatchSemaphore(value: 0)
    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { content, error in
        fetched = content
        fetchError = error
        sem.signal()
    }
    sem.wait()
    if let fetchError {
        throw fetchError
    }
    guard let content = fetched, !content.displays.isEmpty else {
        throw NSError(
            domain: "BsafeCapture", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "No displays found"]
        )
    }

    let primaryID = CGMainDisplayID()
    let filter = rawFilter ?? "primary"
    let selected: [SCDisplay]

    switch filter {
    case "primary":
        if let primary = content.displays.first(where: { $0.displayID == primaryID }) {
            selected = [primary]
        } else {
            // No primary found; fall back to first available display
            selected = [content.displays[0]]  // safe: displays.isEmpty checked above
        }
    case "secondary":
        let nonPrimary = content.displays.filter { $0.displayID != primaryID }
        guard !nonPrimary.isEmpty else {
            throw NSError(
                domain: "BsafeCapture", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "No secondary display found"]
            )
        }
        selected = [nonPrimary[0]]
    case "all":
        selected = content.displays
    default:
        // Numeric display ID
        if let numericID = UInt32(filter) {
            guard let match = content.displays.first(where: { $0.displayID == numericID }) else {
                throw NSError(
                    domain: "BsafeCapture", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "Display with ID \(filter) not found"]
                )
            }
            selected = [match]
        } else {
            throw NSError(
                domain: "BsafeCapture", code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Invalid display filter: \(filter)"]
            )
        }
    }
    return (selected, content)
}

/// SCWindows for the given window-server ids. Fetched after the overlay windows exist, so a capturable
/// overlay can be excluded from our own capture (no feedback loop).
func shareableWindows(withIDs ids: Set<CGWindowID>) -> [SCWindow] {
    if ids.isEmpty { return [] }
    let semaphore = DispatchSemaphore(value: 0)
    var found: [SCWindow] = []
    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: false) { content, _ in
        found = content?.windows.filter { ids.contains($0.windowID) } ?? []
        semaphore.signal()
    }
    semaphore.wait()
    return found
}

/// When set, overlay windows are capturable and our own app is excluded from capture instead.
func isOverlayCapturable() -> Bool {
    return ProcessInfo.processInfo.environment["BSAFE_OVERLAY_CAPTURABLE"] == "1"
}

/// Overlay present-lead in seconds (BSAFE_PRESENT_LEAD_MS env, default 16 ms).
func presentLeadSeconds() -> Double {
    if let s = ProcessInfo.processInfo.environment["BSAFE_PRESENT_LEAD_MS"], let ms = Double(s) {
        return ms / 1000.0
    }
    return 0.016
}
