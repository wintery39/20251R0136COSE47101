from cragmm_search.search import UnifiedSearchPipeline

# initiate image search API only, web search API is not enabled for Task 1
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
)


import requests
from PIL import Image

import time

image_li = []

image_url = "image.png"
origin_image = Image.open(image_url)
# image = ImageLoader(image_url).get_image()
image_li.append(origin_image)

width, height = origin_image.size

# 중간 지점
mid_x = width // 2
mid_y = height // 2

# 각 영역 자르기 (crop의 인자는 (left, upper, right, lower))
divide_coordinates = [
    (0, 0, mid_x, mid_y),  # 왼쪽 위
    (mid_x, 0, width, mid_y),  # 오른쪽 위
    (0, mid_y, mid_x, height),  # 왼쪽 아래
    (mid_x, mid_y, width, height)  # 오른쪽 아래
]

for x, y, w, h in divide_coordinates:
    cropped_image = origin_image.crop((x, y, w, h))
    image_li.append(cropped_image)

entities = dict()
start_time = time.time()
print("start time: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
for image in image_li:
    response = search_pipeline(image, k = 25)
    assert response is not None, "No results found"

    for result in response:
        print(result)
    for result in response:
        for entity in result['entities']:
            entities.update({entity["entity_name"]: entity["entity_attributes"]})
print("end time: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(f"Time taken for search: {time.time() - start_time:.2f} seconds")

# for entity_name, attributes in entities.items():
#     print(f"Entity Name: {entity_name}")
#     print()  # 줄바꿈
