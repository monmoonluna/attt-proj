import AppKit

let output = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "docs/assets"
try FileManager.default.createDirectory(atPath: output, withIntermediateDirectories: true)

let navy = NSColor(calibratedRed: 0.08, green: 0.15, blue: 0.27, alpha: 1)
let blue = NSColor(calibratedRed: 0.13, green: 0.39, blue: 0.75, alpha: 1)
let teal = NSColor(calibratedRed: 0.04, green: 0.53, blue: 0.53, alpha: 1)
let orange = NSColor(calibratedRed: 0.92, green: 0.49, blue: 0.17, alpha: 1)
let gray = NSColor(calibratedRed: 0.42, green: 0.47, blue: 0.54, alpha: 1)
let pale = NSColor(calibratedRed: 0.96, green: 0.97, blue: 0.99, alpha: 1)
let lightBlue = NSColor(calibratedRed: 0.89, green: 0.94, blue: 1.00, alpha: 1)
let lightTeal = NSColor(calibratedRed: 0.88, green: 0.96, blue: 0.95, alpha: 1)
let lightOrange = NSColor(calibratedRed: 1.00, green: 0.94, blue: 0.87, alpha: 1)
let border = NSColor(calibratedRed: 0.81, green: 0.85, blue: 0.90, alpha: 1)

func text(_ value: String, x: CGFloat, y: CGFloat, width: CGFloat, size: CGFloat = 25, color: NSColor = navy, bold: Bool = false, align: NSTextAlignment = .center) {
    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = align
    paragraph.lineSpacing = 4
    let font = bold ? NSFont.boldSystemFont(ofSize: size) : NSFont.systemFont(ofSize: size)
    let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: color, .paragraphStyle: paragraph]
    let string = NSAttributedString(string: value, attributes: attrs)
    let bounds = string.boundingRect(with: NSSize(width: width, height: 2000), options: [.usesLineFragmentOrigin, .usesFontLeading])
    string.draw(in: NSRect(x: x, y: y - bounds.height, width: width, height: bounds.height + 4))
}

func box(_ r: NSRect, fill: NSColor, stroke: NSColor = border, radius: CGFloat = 20) {
    let path = NSBezierPath(roundedRect: r, xRadius: radius, yRadius: radius)
    fill.setFill(); path.fill()
    stroke.setStroke(); path.lineWidth = 2; path.stroke()
}

func line(_ a: NSPoint, _ b: NSPoint, color: NSColor = gray, width: CGFloat = 4, dashed: Bool = false, arrow: Bool = true) {
    let p = NSBezierPath()
    p.move(to: a); p.line(to: b)
    if dashed { p.setLineDash([10, 9], count: 2, phase: 0) }
    color.setStroke(); p.lineWidth = width; p.stroke()
    if arrow {
        let angle = atan2(b.y-a.y, b.x-a.x)
        let length: CGFloat = 15
        let wing: CGFloat = 0.48
        let tip = NSBezierPath()
        tip.move(to: b)
        tip.line(to: NSPoint(x: b.x-length*cos(angle-wing), y: b.y-length*sin(angle-wing)))
        tip.move(to: b)
        tip.line(to: NSPoint(x: b.x-length*cos(angle+wing), y: b.y-length*sin(angle+wing)))
        tip.lineWidth = width; tip.stroke()
    }
}

func image(_ name: String, width: Int = 1600, height: Int = 1000, draw: () -> Void) throws {
    guard let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0), let context = NSGraphicsContext(bitmapImageRep: rep) else { fatalError("Cannot create image") }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    NSColor.white.setFill(); NSRect(x: 0, y: 0, width: width, height: height).fill()
    draw()
    context.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    guard let data = rep.representation(using: .png, properties: [:]) else { fatalError("Cannot encode image") }
    try data.write(to: URL(fileURLWithPath: "\(output)/\(name).png"))
}

try image("quy-trinh-nids", height: 1110) {
    text("QUY TRÌNH NGHIÊN CỨU NIDS", x: 100, y: 1060, width: 1400, size: 42, bold: true)
    text("Từ dữ liệu mạng đến so sánh hiệu quả chọn đặc trưng", x: 100, y: 1005, width: 1400, size: 25, color: gray)

    box(NSRect(x: 125, y: 805, width: 1350, height: 145), fill: pale)
    text("DỮ LIỆU ĐẦU VÀO", x: 160, y: 918, width: 1280, size: 24, color: blue, bold: true)
    text("NSL-KDD  •  UNSW-NB15  •  CICIDS2017", x: 160, y: 870, width: 1280, size: 32, bold: true)
    line(NSPoint(x: 800, y: 805), NSPoint(x: 800, y: 748))

    box(NSRect(x: 125, y: 600, width: 1350, height: 145), fill: lightBlue)
    text("TIỀN XỬ LÝ", x: 160, y: 713, width: 1280, size: 24, color: blue, bold: true)
    text("Làm sạch  →  mã hóa  →  chuẩn hóa  →  train / test", x: 160, y: 665, width: 1280, size: 31, bold: true)
    line(NSPoint(x: 800, y: 600), NSPoint(x: 800, y: 540))

    box(NSRect(x: 125, y: 345, width: 600, height: 190), fill: lightTeal)
    text("CHỌN ĐẶC TRƯNG", x: 165, y: 495, width: 520, size: 24, color: teal, bold: true)
    text("GA hoặc PSO", x: 165, y: 447, width: 520, size: 34, bold: true)
    text("Tối ưu F1-macro và số đặc trưng", x: 165, y: 396, width: 520, size: 23)

    box(NSRect(x: 875, y: 345, width: 600, height: 190), fill: pale)
    text("BASELINE", x: 915, y: 495, width: 520, size: 24, color: gray, bold: true)
    text("Giữ toàn bộ đặc trưng", x: 915, y: 447, width: 520, size: 32, bold: true)
    text("Cùng classifier và tập test", x: 915, y: 396, width: 520, size: 23)
    line(NSPoint(x: 800, y: 540), NSPoint(x: 425, y: 540), arrow: false)
    line(NSPoint(x: 425, y: 540), NSPoint(x: 425, y: 535))
    line(NSPoint(x: 800, y: 540), NSPoint(x: 1175, y: 540), arrow: false)
    line(NSPoint(x: 1175, y: 540), NSPoint(x: 1175, y: 535))

    line(NSPoint(x: 425, y: 345), NSPoint(x: 425, y: 290))
    line(NSPoint(x: 1175, y: 345), NSPoint(x: 1175, y: 290))
    box(NSRect(x: 125, y: 120, width: 1350, height: 165), fill: lightOrange)
    text("ĐÁNH GIÁ VÀ SO SÁNH", x: 160, y: 250, width: 1280, size: 25, color: orange, bold: true)
    text("F1-macro  •  Accuracy  •  số đặc trưng  •  thời gian", x: 160, y: 200, width: 1280, size: 30, bold: true)
    text("Đã triển khai: pipeline + GA/PSO + baseline. Hybrid Filter → GA/PSO là bước kế tiếp.", x: 125, y: 80, width: 1350, size: 21, color: gray)
}

try image("co-che-chon-dac-trung", height: 960) {
    text("GA VÀ PSO CHỌN ĐẶC TRƯNG NHƯ THẾ NÀO?", x: 85, y: 910, width: 1430, size: 39, bold: true)
    text("Mỗi phương án là một mặt nạ nhị phân: 1 = giữ, 0 = bỏ đặc trưng", x: 85, y: 855, width: 1430, size: 25, color: gray)
    let bits = [1, 0, 1, 1, 0, 0, 1, 0]
    for (i, bit) in bits.enumerated() {
        let x = CGFloat(285 + i*130)
        box(NSRect(x: x, y: 650, width: 100, height: 100), fill: bit == 1 ? lightTeal : pale, stroke: bit == 1 ? teal : border, radius: 14)
        text(String(bit), x: x, y: 722, width: 100, size: 39, color: bit == 1 ? teal : gray, bold: true)
        text("F\(i+1)", x: x, y: 632, width: 100, size: 20, color: gray)
    }
    line(NSPoint(x: 800, y: 615), NSPoint(x: 800, y: 555))
    box(NSRect(x: 170, y: 390, width: 1260, height: 160), fill: lightBlue)
    text("HÀM ĐÁNH GIÁ HIỆN TẠI", x: 210, y: 520, width: 1180, size: 23, color: blue, bold: true)
    text("Fitness = 0,9 × F1-macro − 0,1 × tỷ lệ đặc trưng được chọn", x: 210, y: 466, width: 1180, size: 32, bold: true)
    text("F1 cao hơn làm fitness tăng; chọn nhiều đặc trưng làm fitness giảm.", x: 210, y: 422, width: 1180, size: 21, color: gray)
    line(NSPoint(x: 800, y: 390), NSPoint(x: 800, y: 335))
    box(NSRect(x: 170, y: 105, width: 570, height: 225), fill: lightTeal)
    text("GENETIC ALGORITHM", x: 205, y: 290, width: 500, size: 25, color: teal, bold: true)
    text("Chọn lọc", x: 205, y: 238, width: 500, size: 28, bold: true)
    text("Lai ghép  →  đột biến", x: 205, y: 192, width: 500, size: 27)
    text("Giữ nghiệm tốt qua các thế hệ", x: 205, y: 145, width: 500, size: 21, color: gray)
    box(NSRect(x: 860, y: 105, width: 570, height: 225), fill: lightOrange)
    text("BINARY PSO", x: 895, y: 290, width: 500, size: 25, color: orange, bold: true)
    text("Cập nhật vận tốc", x: 895, y: 238, width: 500, size: 28, bold: true)
    text("Vận tốc  →  xác suất bit = 1", x: 895, y: 192, width: 500, size: 25)
    text("Dựa vào nghiệm cá nhân và toàn đàn", x: 895, y: 145, width: 500, size: 21, color: gray)
    line(NSPoint(x: 800, y: 335), NSPoint(x: 455, y: 335), arrow: false)
    line(NSPoint(x: 455, y: 335), NSPoint(x: 455, y: 330))
    line(NSPoint(x: 800, y: 335), NSPoint(x: 1145, y: 335), arrow: false)
    line(NSPoint(x: 1145, y: 335), NSPoint(x: 1145, y: 330))
}

try image("ket-qua-f1-va-so-dac-trung", height: 1000) {
    text("KẾT QUẢ THỬ NGHIỆM BƯỚC ĐẦU", x: 95, y: 950, width: 1410, size: 42, bold: true)
    text("Decision Tree • phân loại nhị phân • F1-macro trên tập test", x: 95, y: 895, width: 1410, size: 25, color: gray)
    let rows: [(String, Int, Int, Int, Double, Double, Double)] = [
        ("NSL-KDD",39,6,15,0.7936,0.7351,0.7708),
        ("UNSW-NB15",42,4,15,0.8553,0.8274,0.8569),
        ("CICIDS2017",52,5,22,0.9975,0.9959,0.9981)
    ]
    let startYs: [CGFloat] = [740, 495, 250]
    for (i, row) in rows.enumerated() {
        let y = startYs[i]
        text(row.0, x: 95, y: y+100, width: 340, size: 30, bold: true, align: .left)
        let names = ["Toàn bộ", "GA", "PSO"]
        let counts = [row.1, row.2, row.3]
        let scores = [row.4, row.5, row.6]
        let colors = [gray, blue, teal]
        for j in 0..<3 {
            let yy = y - CGFloat(j)*56
            text("\(names[j])  \(counts[j])/\(row.1)", x: 95, y: yy+25, width: 300, size: 22, color: colors[j], bold: true, align: .left)
            box(NSRect(x: 420, y: yy-12, width: 875, height: 30), fill: pale, stroke: pale, radius: 12)
            box(NSRect(x: 420, y: yy-12, width: 875*CGFloat(scores[j]), height: 30), fill: colors[j], stroke: colors[j], radius: 12)
            text(String(format:"%.4f", scores[j]), x: 1320, y: yy+22, width: 180, size: 23, bold: true, align: .left)
        }
    }
    text("Thanh biểu diễn F1-macro trên thang 0–1; con số trước thanh là số đặc trưng được giữ.", x: 95, y: 67, width: 1410, size: 21, color: gray, align: .left)
}
