import 'dart:async';
import 'dart:typed_data';
import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:web_socket_channel/io.dart';
import 'package:image/image.dart' as img;

class CameraStreamPage extends StatefulWidget {
  final String ip;
  final CameraController cameraController;

  CameraStreamPage({required this.cameraController, this.ip = '10.0.2.2:8765'});

  @override
  _CameraStreamPageState createState() => _CameraStreamPageState();
}

class _CameraStreamPageState extends State<CameraStreamPage> {
  final StreamController<Uint8List> _imageStreamController = StreamController.broadcast();
  Uint8List? _receivedImage;
  late IOWebSocketChannel _channel;
  bool _isStreaming = false;
  int _lastSent = 0;

  late CameraController _cameraControllerSource;

  @override
  void initState() {
    super.initState();

    _cameraControllerSource = widget.cameraController;

    _connectWebSocket();

    _startImageStreamIfNeeded();
  }

  void _connectWebSocket() {
    _channel = IOWebSocketChannel.connect('ws://${widget.ip}/backend/upload');

    _channel.stream.listen((data) {
      if (data is Uint8List) {
        _receivedImage = data;
        _imageStreamController.add(data);
      }
    }, onError: (error) {
      print("WebSocket error: $error");
    });
  }

  Future<Uint8List?> convertYUV420toJPEG(CameraImage image) async {
      try {
        final int width = image.width;
        final int height = image.height;

        final int yRowStride = image.planes[0].bytesPerRow;
        final int uvRowStride = image.planes[1].bytesPerRow;
        final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

        final img.Image rgbImage = img.Image(width: width, height: height);

        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final int uvX = x ~/ 2;
            final int uvY = y ~/ 2;

            final int uvIndex = uvY * uvRowStride + uvX * uvPixelStride;

            final int yIndex = y * yRowStride + x;

            final int yp = image.planes[0].bytes[yIndex];
            final int up = image.planes[1].bytes[uvIndex];
            final int vp = image.planes[2].bytes[uvIndex];

            int r = (yp + 1.370705 * (vp - 128)).round();
            int g = (yp - 0.698001 * (vp - 128) - 0.337633 * (up - 128)).round();
            int b = (yp + 1.732446 * (up - 128)).round();

            rgbImage.setPixelRgba(
              x,
              y,
              r.clamp(0, 255),
              g.clamp(0, 255),
              b.clamp(0, 255),
              255,
            );
          }
        }

        return Uint8List.fromList(img.encodeJpg(rgbImage, quality: 50));
      } catch (e) {
        print('Conversion error: $e');
        return null;
      }
    }


  void _startImageStreamIfNeeded() {
    if (!_isStreaming && _cameraControllerSource.value.isInitialized) {
      _cameraControllerSource.startImageStream((CameraImage image) async {
        final now = DateTime.now().millisecondsSinceEpoch;
        if (now - _lastSent < 50) return; // Throttle to 20fps
        _lastSent = now;

        try {
          final jpeg = await convertYUV420toJPEG(image);
          if (jpeg != null && _channel.sink != null) {
            _channel.sink.add(jpeg);
          }
        } catch (e) {
          print('Encoding error: $e');
        }
      });
      _isStreaming = true;
    }
  }

  @override
  void dispose() {
    if (_isStreaming) {
      _cameraControllerSource.stopImageStream();
      _isStreaming = false;
    }
    // Don't dispose the controller here, HomePage owns it.
    _imageStreamController.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraControllerSource.value.isInitialized) {
      return Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: Text('Camera Streamer', style: TextStyle(
          color: Colors.white
        ),),
        backgroundColor: Theme.of(context).primaryColor,
        iconTheme: IconThemeData(color: Colors.white),
      ),
      body: Center(
        child: StreamBuilder<Uint8List>(
          stream: _imageStreamController.stream,
          builder: (context, snapshot) {
            if (!snapshot.hasData) {
              return const CircularProgressIndicator();
            }

            if (snapshot.connectionState == ConnectionState.done) {
              return const Center(
                child: Text("Connection Closed !"),
              );
            }
            return Image.memory(
              snapshot.data!,
              gaplessPlayback: true,
              excludeFromSemantics: true,
            );
          },
        ),
      ),
    );
  }
}
