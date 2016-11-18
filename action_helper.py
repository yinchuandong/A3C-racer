# encoding=utf-8

action_left = (True, False, True, False)
action_right = (False, True, True, False)
action_faster = (False, False, True, False)
action_id_dict = dict()
action_id_dict[action_left] = 0
action_id_dict[action_right] = 1
action_id_dict[action_faster] = 2
id_action_dict = dict()
id_action_dict[0] = action_left
id_action_dict[1] = action_right
id_action_dict[2] = action_faster


def encode_action(left, right, faster, slower):
    if left and right:
        raise ValueError("Invalid action, cannot press both left and right")
    if faster and slower:
        raise ValueError("Invalid action, cannot press both faster and slower")
    return action_id_dict[(left, right, faster, slower)]


def decode_action(action_id):
    left, right, faster, slower = id_action_dict[action_id]
    return {
        'keyLeft': left,
        'keyRight': right,
        'keyFaster': faster,
        'keySlower': slower
    }


if __name__ == '__main__':
    id = encode_action(False, True, True, False)
    json = decode_action(2)
    # json = decode_action(id)
    print id
    print json
