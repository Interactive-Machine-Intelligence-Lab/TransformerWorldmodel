import torch
import imageio
import cv2


data_src = 'output/2023-05-17_13-46-14/media/episodes/train/best_episode_59_epoch_54.pt'
data = torch.load(data_src)

obs = data['observations'].permute(0, 2, 3, 1).numpy().copy()
rwd = data['rewards'].numpy().copy()


res = []
total = 0

for o, r in zip(obs, rwd):
    o = cv2.resize(o, (256, 256))
    o = cv2.putText(o, 'reward : ' + str(round(r, 5)), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    total += r
    o = cv2.putText(o, 'total : ' + str(round(total, 5)), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    res.append(o)


imageio.mimsave('test.gif', res)