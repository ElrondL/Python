#import numpy as np

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

    # advanced solution
    # students = [[input(),float(input())] for i in range(int(input()))]
    # second_highest = sorted(set(j for i,j in students))[1]
    # print("\n".join(sorted(i for i,j in students if j==second_highest)))


if __name__ == '__main__':
    N = int(input())
    # N = 4
    name = ['name']*N
    score = [1.5]*N
    for i in range(N):
        name[i] = input()
        score[i] = float(input())

    score_ID = argsort(score)
    # print(score_ID)
    for j in range(1,N):
        if score[score_ID[j]] != score[score_ID[0]]:
            key_ID = [int(score_ID[j])]
            id = j
            break
    # print(key_ID)
    # print(id)
    # print(name, score)
    # name_list = [name[int(key_ID[0])]]
    # print(name_list)
    # print(score[score_ID[2]])
    # print(score[score_ID[1]])
    # print(score_ID[2])
    for i in range(id+1, N):
        # print(i)
        if score[score_ID[i]] == score[score_ID[id]]:
            key_ID.append(int(score_ID[i]))

    # print(key_ID)
    # print(name_list)
    # key_ID = [int(s) for s in key_ID]
    # print(key_ID)
    # print(np.array(key_ID))
    name_list = [name[id] for id in key_ID]

    print(*sorted(name_list), sep="\n")

