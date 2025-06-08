from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import sqlite3
from contextlib import contextmanager
from ultralytics import YOLO

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 数据库配置
DATABASE = 'data/detections.db'

# 病害类别映射
DISEASE_CLASSES = {
    0: '角斑病',
    1: '炭疽病',
    2: '花腐病',
    3: '灰霉病',
    4: '健康花朵',
    5: '健康果实',
    6: '健康叶片',
    7: '叶斑病',
    8: '果实白粉病',
    9: '叶片白粉病'
}

# 加载模型
model = YOLO('data/best.mlpackage')

@contextmanager
def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()

def init_db():
    """初始化数据库"""
    try:
        with get_db() as db:
            # 创建检测记录表
            db.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    result_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    zone_id TEXT NOT NULL,
                    plant_id TEXT NOT NULL,
                    original_image TEXT NOT NULL,
                    annotated_image TEXT NOT NULL,
                    conf REAL NOT NULL,
                    iou REAL NOT NULL
                )
            ''')
            
            # 创建检测项表
            db.execute('''
                CREATE TABLE IF NOT EXISTS detection_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id TEXT NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox TEXT NOT NULL,
                    FOREIGN KEY (result_id) REFERENCES detections (result_id)
                )
            ''')
            db.commit()
            logger.info("数据库初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 获取检测参数
            conf = float(request.form.get('conf', 0.25))
            iou = float(request.form.get('iou', 0.7))
            zone_id = request.form.get('zone_id', '')
            plant_id = request.form.get('plant_id', '')
            
            # 执行检测
            results = model(
                source=filepath,
                conf=conf,
                iou=iou,
                half=True,
                save=True
            )
            
            # 保存标注后的图片
            annotated_img = results[0].plot()
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], f'annotated_{filename}')
            cv2.imwrite(annotated_path, annotated_img)
            
            # 生成结果ID
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            result_id = f"{zone_id}_{plant_id}_{timestamp}"
            
            # 保存检测结果到数据库
            with get_db() as db:
                # 保存主记录
                db.execute('''
                    INSERT INTO detections 
                    (result_id, timestamp, zone_id, plant_id, original_image, annotated_image, conf, iou)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    zone_id,
                    plant_id,
                    f'/static/uploads/{filename}',
                    f'/static/uploads/annotated_{filename}',
                    conf,
                    iou
                ))
                
                # 保存检测项
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = DISEASE_CLASSES.get(class_id, f'未知类别_{class_id}')
                        bbox_json = json.dumps(box.xyxy[0].tolist())
                        db.execute('''
                            INSERT INTO detection_items 
                            (result_id, class_id, class_name, confidence, bbox)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            result_id,
                            class_id,
                            class_name,
                            float(box.conf[0]),
                            bbox_json
                        ))
                
                db.commit()
                logger.info(f"检测结果已保存到数据库，ID: {result_id}")
            
            # 获取检测结果用于返回
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    detection = {
                        'class': class_id,  # 返回数字类别
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    }
                    detections.append(detection)
            
            return jsonify({
                'success': True,
                'result_id': result_id,
                'original_image': f'/static/uploads/{filename}',
                'annotated_image': f'/static/uploads/annotated_{filename}',
                'detections': detections
            })
        except Exception as e:
            logger.error(f"处理上传文件时出错: {str(e)}")
            return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500
    
    return jsonify({'error': '不支持的文件类型'}), 400

@app.route('/get_detections')
def get_detections():
    """获取所有检测结果"""
    try:
        with get_db() as db:
            detections = {}
            
            # 获取所有检测记录
            cursor = db.execute('''
                SELECT * FROM detections
                ORDER BY timestamp DESC
            ''')
            
            for row in cursor:
                result = dict(row)
                result_id = result['result_id']
                
                # 获取该记录的所有检测项
                items_cursor = db.execute('''
                    SELECT class_id, class_name, confidence, bbox
                    FROM detection_items
                    WHERE result_id = ?
                ''', (result_id,))
                
                detections_list = []
                for item in items_cursor:
                    try:
                        bbox = json.loads(item['bbox'])
                        detection_item = {
                            'class': item['class_id'],
                            'class_name': item['class_name'],
                            'confidence': item['confidence'],
                            'bbox': bbox
                        }
                        logger.info(f"检测项数据: {detection_item}")  # 添加日志
                        detections_list.append(detection_item)
                    except json.JSONDecodeError as e:
                        logger.error(f"解析bbox JSON失败: {str(e)}")
                        continue
                
                result['detections'] = detections_list
                detections[result_id] = result
                logger.info(f"检测记录 {result_id} 的检测项数量: {len(detections_list)}")  # 添加日志
            
            logger.info(f"成功获取检测结果，共 {len(detections)} 条记录")
            return jsonify(detections)
    except Exception as e:
        logger.error(f"获取检测结果时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 初始化数据库
init_db()

if __name__ == '__main__':
    logger.info("应用启动成功")
    app.run(debug=True) 