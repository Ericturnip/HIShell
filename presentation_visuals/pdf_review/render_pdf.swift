import AppKit
import PDFKit

let args = CommandLine.arguments
guard args.count >= 3 else {
    fputs("usage: render_pdf.swift input.pdf output_dir\n", stderr)
    exit(2)
}

let pdfURL = URL(fileURLWithPath: args[1])
let outURL = URL(fileURLWithPath: args[2])
try FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)

guard let doc = PDFDocument(url: pdfURL) else {
    fputs("could not open PDF\n", stderr)
    exit(1)
}

for idx in 0..<doc.pageCount {
    guard let page = doc.page(at: idx) else { continue }
    let bounds = page.bounds(for: .mediaBox)
    let scale: CGFloat = 2.0
    let size = NSSize(width: bounds.width * scale, height: bounds.height * scale)
    let image = NSImage(size: size)
    image.lockFocus()
    NSColor.white.setFill()
    NSRect(origin: .zero, size: size).fill()
    let context = NSGraphicsContext.current!.cgContext
    context.saveGState()
    context.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: context)
    context.restoreGState()
    image.unlockFocus()

    guard let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:]) else {
        continue
    }
    let path = outURL.appendingPathComponent(String(format: "page_%02d.png", idx + 1))
    try png.write(to: path)
    print(path.path)
}
