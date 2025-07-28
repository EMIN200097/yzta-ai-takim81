import 'package:flutter/material.dart';
import 'camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:camera/camera.dart';

late List<CameraDescription> cameras;

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final TextEditingController ipController = TextEditingController(text: "10.0.2.2:8765");
  CameraController? _cameraController;
  bool _cameraInitialized = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    if (status != PermissionStatus.granted) {
      throw Exception('Camera permission not granted');
    }
  }

  Future<void> _initCamera() async {
    await _requestCameraPermission();
    cameras = await availableCameras();
    final controller = CameraController(
      cameras[0],
      ResolutionPreset.low,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await controller.initialize();

    if (mounted) {
      setState(() {
        _cameraController = controller;
        _cameraInitialized = true;
      });
    }
  }

  @override
  void dispose() {
    ipController.dispose();
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Home',
          style: TextStyle(
            color: Colors.white
          ),
        ),
        iconTheme: IconThemeData(color: Colors.white),
        backgroundColor: Theme.of(context).primaryColor
      ),
      drawer: Drawer(
        child: Column(
          children: [
            DrawerHeader(
              decoration: BoxDecoration(
                color: Theme.of(context).primaryColor,
              ),
              child: Center(
                child: Text(
                  "Select a mode",
                  style: TextStyle(color: Colors.white, fontSize: 30, fontWeight: FontWeight.bold),
                ),
              ),
            ),
            ListTile(
              leading: Icon(Icons.event_seat),
              title: Text('Sitting Posture Analysis'),
              onTap: () {
                Navigator.pop(context);
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => Scaffold(
                      appBar: AppBar(title: Text("Test Page")),
                      body: Center(child: Text("No error here")),
                    ),
                  ),
                );
              },
            ),
            ListTile(
              leading: Icon(Icons.fitness_center),
              title: Text('Squat Analysis'),
              onTap: () {
                if (!_cameraInitialized || _cameraController == null) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text("Camera not ready yet")),
                  );
                  return;
                }
                Navigator.pop(context);
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => CameraStreamPage(
                      cameraController: _cameraController!,
                      ip: ipController.text,
                    ),
                  ),
                );
              },
            ),
            Spacer(),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Text(
                '© 2025 PoseCore',
                style: TextStyle(color: Colors.grey[600], fontSize: 12),
              ),
            ),
          ],
        ),
      ),
      body: Center(
        child: Padding(
          padding: EdgeInsets.only(top: 100),
          child: Column(
            children: [
              Image.asset('assets/images/posecore_logo.jpg'),
              Container(
                width: 300,
                padding: EdgeInsets.symmetric(horizontal: 16),
                child: TextField(
                  controller: ipController,
                  decoration: InputDecoration(
                    labelText: 'Enter Server IP Address and Port',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    prefixIcon: Icon(Icons.network_wifi),
                    filled: true,
                    fillColor: Colors.grey[200],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
