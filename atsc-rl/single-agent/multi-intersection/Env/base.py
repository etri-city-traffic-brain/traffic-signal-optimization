
class baseEnv():
    def __init__(self, configs):
        self.configs=configs
        '''
        base env
        '''

    def get_state(self):
        '''
        상속을 위한 함수
        return state torch.Tensor(dtype=torch.int)
        '''
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
    
    def collect_state(self):
        raise NotImplementedError

    def get_reward(self):
        '''
        reward function
        Max Pressure based control
        return reward torch.Tensor(dtype=torch.int)
        '''
        raise NotImplementedError

    def _toPhase(self, action):  # action을 해석가능한 phase로 변환
        '''
        right: green signal
        straight: green=1, yellow=x, red=0 <- x is for changing
        left: green=1, yellow=x, red=0 <- x is for changing
        '''
        signal_set = list()
        phase_set=tuple()
        phase = str()
        for _, a in enumerate(action):
            signal_set.append(self._getMovement(a))
        for j,signal in enumerate(signal_set):
            # 1개당
            for i in range(4):  # 4차로
                phase = phase + 'g'+self.configs['numLane']*signal[j][2*i] + \
                    signal[j][2*i+1]+'r'  # 마지막 r은 u-turn
            phase_set+=phase
        print(phase_set)
        return phase_set

    def _toState(self, phase_set):  # env의 phase를 해석불가능한 state로 변환
        state_set=tuple()
        for i,phase in enumerate(phase_set):
            state = torch.zeros(8, dtype=torch.int)
            for i in range(4):  # 4차로
                phase = phase[1:]  # 우회전
                state[i] = self._mappingMovement(phase[0])  # 직진신호 추출
                phase = phase[3:]  # 직전
                state[i+1] = self._mappingMovement(phase[0])  # 좌회전신호 추출
                phase = phase[1:]  # 좌회전
                phase = phase[1:]  # 유턴
            state_set+=state
        return state_set

    def _getMovement(self, num):
        if num == 1:
            return 'G'
        elif num == 0:
            return 'r'
        else:
            return 'y'

    def _mappingMovement(self, movement):
        if movement == 'G':
            return 1
        elif movement == 'r':
            return 0
        else:
            return -1  # error
