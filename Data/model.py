from ultralytics import YOLO
uddoktaModel = YOLO('AI_Model/uddokta_v11.1.pt')
marchentModel = YOLO('AI_Model/marchent_v11.1.pt')

uddoktaModel.to(device=0)
marchentModel.to(device=0)
