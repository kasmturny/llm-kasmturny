import numpy as np
from tqdm import tqdm
import threading


# 假设你的数据已经加载到变量data中
data = np.load('D:\\Exploitation\\All\\llm-kasmturny\\pyorigin\\agent\\bio_ner_finetune\\data\\test.npz', allow_pickle=True)

def process_data(data_chunk):
    input_list = []
    output_list = []
    input_text = []
    for i in tqdm(range(len(data_chunk['words']))):
        word = data_chunk['words'][i]
        label = data_chunk['labels'][i]
        input_list.append([])
        output_list.append([])
        input_text.append(''.join(word))
        for j in range(len(word)):
            input_list[i].append([word[j], j])
            output_list[i].append([word[j], j, label[j]])
    return input_list, output_list, input_text


def thread_function(data_chunk, results, index):
    results[index] = process_data(data_chunk)


def main():
    num_threads = 4  # 你可以根据你的需求设置线程数
    words = data['words']
    labels = data['labels']

    # 将数据分割成多个块，每个线程处理一个块
    chunk_size = len(words) // num_threads
    threads = []
    results = [None] * num_threads

    for i in range(num_threads):
        start_index = i * chunk_size
        end_index = None if i == num_threads - 1 else (i + 1) * chunk_size
        data_chunk = {'words': words[start_index:end_index], 'labels': labels[start_index:end_index]}

        thread = threading.Thread(target=thread_function, args=(data_chunk, results, i))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 合并结果
    input_lists, output_lists, input_texts = zip(*results)
    input_list = [item for sublist in input_lists for item in sublist]
    output_list = [item for sublist in output_lists for item in sublist]
    input_text = [item for sublist in input_texts for item in sublist]
    return input_list, output_list, input_text

    # 现在input_list, output_list, input_text包含了所有处理后的数据


if __name__ == "__main__":
    import time
    start_time = time.time()
    input_list, output_list, input_text = process_data(data)
    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    input_list1, output_list1, input_text1 = main()
    print("--- %s seconds ---" % (time.time() - start_time))

    if input_list == input_list1 and output_list == output_list1 and input_text == input_text1:print('ok')
    print('速度差距十分之大，一个29s,一个0.2s,因为是io密集型，cpu使用率不高，所以多线程可以显著提升效果，如果是cpu密集型，就不会有这么大的效果')

    # result
    """
    100%|██████████| 1343/1343 [00:23<00:00, 56.73it/s]
    --- 23.717585563659668 seconds ---
    100%|██████████| 335/335 [00:00<00:00, 9571.27it/s]
    100%|██████████| 335/335 [00:00<00:00, 83746.09it/s]
    100%|██████████| 335/335 [00:00<00:00, 37210.13it/s]
    100%|██████████| 338/338 [00:00<00:00, 30731.50it/s]
    --- 0.07999968528747559 seconds ---
    ok
    速度差距十分之，一个29s,一个0.2s,因为是io密集型，cpu使用率不高，所以多线程可以显著提升效果，如果是cpu密集型，就不会有这么大的效果
    """



