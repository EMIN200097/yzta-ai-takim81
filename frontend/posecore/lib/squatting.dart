import 'dart:async';
import 'dart:typed_data';
import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:web_socket_channel/io.dart';
import 'package:image/image.dart' as img;
import 'package:audioplayers/audioplayers.dart';

class SquattingPage extends StatefulWidget {
  final String ip;
  final CameraController cameraController;

  SquattingPage({required this.cameraController, this.ip = '10.0.2.2:8765'});

  @override
  _SquattingPageState createState() => _SquattingPageState();
}

class _SquattingPageState extends State<SquattingPage> {
  final _player = AudioPlayer();
  final StreamController<Uint8List> _imageStreamController = StreamController.broadcast();
  Uint8List? _receivedImage;
  int _score = 3000; // Sınce we use dtw for calculating score, This scales from ~3000 to ~1000 worst to best.
  bool _isProcessingFrame = false;
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
    _channel = IOWebSocketChannel.connect('ws://${widget.ip}/backend/squatting/upload');

    _channel.stream.listen((data) {
      if (data is Uint8List) {
        setState(() {
          _receivedImage = data;
        });
      } else if(data is String) {
        final jsonData = jsonDecode(data);
        setState(() {
          _score = jsonData['score'];  // <-- update inside setState!
        });
        if(_score! > 1500) {
          _player.play(AssetSource('sounds/warning.wav'));
        } else {
          _player.stop();
        }
      }
    }, onError: (error) {
      print("WebSocket error: $error");
    });
  }

  void _startImageStreamIfNeeded() {
    _cameraControllerSource.startImageStream((CameraImage image) async {
      _isStreaming = true;
      if (_isProcessingFrame) return;

      final now = DateTime.now().millisecondsSinceEpoch;
      if (now - _lastSent < 100) return;

      _lastSent = now;
      _isProcessingFrame = true;

      try {
        final width = image.width;
        final height = image.height;

        final yPlane = image.planes[0].bytes;
        final uPlane = image.planes[1].bytes;
        final vPlane = image.planes[2].bytes;

        // Create a single Uint8List buffer of size width*height*3/2
        final yuv420p = Uint8List(width * height * 3 ~/ 2);

        // Copy Y plane
        yuv420p.setRange(0, width * height, yPlane);

        // Copy U plane after Y
        yuv420p.setRange(width * height, width * height + (width * height) ~/ 4, uPlane);

        // Copy V plane after U
        yuv420p.setRange(width * height + (width * height) ~/ 4, width * height * 3 ~/ 2, vPlane);

        // Send buffer over websocket
        _channel.sink.add(yuv420p);
      } catch (e) {
        print('Encoding error: $e');
      } finally {
        _isProcessingFrame = false;
      }
    });
  }

  @override
  void dispose() {
    if (_isStreaming) {
      _cameraControllerSource.stopImageStream();
      _isStreaming = false;
    }
    // Don't dispose the controller here, HomePage owns it.
    _imageStreamController.close();
    _player.stop();
    _channel.sink.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraControllerSource.value.isInitialized) {
      return Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: Text('Squat Posture Analysis', style: TextStyle(
          color: Colors.white
        ),),
        backgroundColor: Theme.of(context).primaryColor,
        iconTheme: IconThemeData(color: Colors.white),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
        Flexible(
          fit: FlexFit.loose,
          child: _receivedImage == null
            ? CircularProgressIndicator()
            : Image.memory(
                _receivedImage!,
                gaplessPlayback: true,
                excludeFromSemantics: true,
              ),
        ),
        const SizedBox(height: 16),
        Center(
          child: Container(
            padding: const EdgeInsets.symmetric(
              vertical: 6,
              horizontal: 16,
            ),
            decoration: BoxDecoration(
              color: _score! < 1500 ? Colors.green : Colors.red,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
          color: Colors.black54,
            width: 2,
              ),
            ),
            child: Text(
              "$_score",
              style: const TextStyle(fontSize: 32, color: Colors.white),
            ),
          ),
        ),
        const SizedBox(height: 64),
          ],
        ),
      ),
    );
  }
}
