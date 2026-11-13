JSON 주석 형식 (split/train.json, split/val.json):

[
  {
    "image": "relative/path/to/image.png",
    "center": [cx, cy],
    "scale": 1.2,
    "joints": [
      [x0, y0],  // 0: Right foot
      [x1, y1],  // 1: Right knee
      [x2, y2],  // 2: Right hip
      [x3, y3],  // 3: Left hip
      [x4, y4],  // 4: Left knee
      [x5, y5],  // 5: Left foot
      [x6, y6],  // 6: Pelvis
      [x7, y7],  // 7: Spine naval
      [x8, y8],  // 8: Spine chest
      [x9, y9],  // 9: Neck base
      [x10,y10], // 10: Center head
      [x11,y11], // 11: Right hand
      [x12,y12], // 12: Right elbow
      [x13,y13], // 13: Right shoulder
      [x14,y14], // 14: Left shoulder
      [x15,y15], // 15: Left elbow
      [x16,y16]  // 16: Left hand
    ],
    "joints_vis": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  },
  ...
]

필드 설명:
  image      : DATA_ROOT 기준 상대경로 (역슬래시/슬래시 모두 허용)
  center     : 사람 바운딩박스 중심 픽셀 좌표
  scale      : max(bbox_w, bbox_h) / 200  (affine crop 배율)
  joints     : NIA 17관절 원본 이미지 픽셀 좌표
  joints_vis : 1=가시, 0=불가시(occluded / outside)

source 데이터 변환:
  source/2. 키포인트 객체 인식/UniPose/split/test.json 이
  이 형식과 거의 동일하므로 참고하면 된다.
  source/2. 키포인트 객체 인식/UniPose/lib/utils/read_data_from_nia.py 의
  get_one_gt() 함수로 XML → 이 JSON 형식으로 변환 가능.
