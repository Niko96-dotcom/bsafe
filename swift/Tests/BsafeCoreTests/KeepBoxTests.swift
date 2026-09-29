import XCTest
@testable import BsafeCore

final class KeepBoxTests: XCTestCase {
    /// Keep-only boxes refresh a track they match but never start a track.
    func testKeepBoxKeepsTrackAliveAndNeverCreatesOne() {
        var config = TrackerConfig()
        config.persistPasses = 2
        config.minPersistSeconds = 0.0
        let base = makeTexture(width: 640, height: 400, seed: 0x5151)
        let tracker = DisplayTracker(frameWidth: 640, frameHeight: 400, config: config)
        let box = TrackBox(x: 200, y: 150, w: 160, h: 100)
        tracker.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tracker.applyDetections(seq: 0, boxes: [box]))
        XCTAssertEqual(tracker.tracks.count, 1)

        // Five passes with only a keep box on the object: the track survives
        // (persistPasses 2 would drop it after two plain misses).
        for i in 1...5 {
            tracker.ingestFrame(seq: UInt32(i), pts: Double(i) * 0.1, pyramid: pyr(base))
            XCTAssertTrue(tracker.applyDetections(seq: UInt32(i), boxes: [], keepBoxes: [box]))
        }
        XCTAssertEqual(tracker.tracks.count, 1)
        XCTAssertEqual(tracker.tracks[0].misses, 0)

        // A keep box elsewhere starts nothing, and the unmatched track misses.
        let elsewhere = TrackBox(x: 20, y: 20, w: 50, h: 50)
        tracker.ingestFrame(seq: 6, pts: 0.6, pyramid: pyr(base))
        XCTAssertTrue(tracker.applyDetections(seq: 6, boxes: [], keepBoxes: [elsewhere]))
        XCTAssertEqual(tracker.tracks.count, 1)
        XCTAssertEqual(tracker.tracks[0].misses, 1)
    }

    /// A strong detection wins its track before keep boxes are matched, so a
    /// keep box overlapping the same track cannot push the strong one into a
    /// duplicate track.
    func testStrongMatchesFirst() {
        let base = makeTexture(width: 640, height: 400, seed: 0x5252)
        let tracker = DisplayTracker(frameWidth: 640, frameHeight: 400)
        let box = TrackBox(x: 200, y: 150, w: 160, h: 100)
        tracker.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tracker.applyDetections(seq: 0, boxes: [box]))
        tracker.ingestFrame(seq: 1, pts: 0.1, pyramid: pyr(base))
        let shifted = TrackBox(x: 230, y: 160, w: 160, h: 100)
        XCTAssertTrue(tracker.applyDetections(seq: 1, boxes: [shifted], keepBoxes: [box]))
        XCTAssertEqual(tracker.tracks.count, 1)
    }
}
