#pragma once
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

#include <winsock2.h>
#pragma comment(lib,"WS2_32")
#include <ws2tcpip.h> 
#pragma comment(lib, "ws2_32.lib") 

class CNet
{
public:
	CNet(void);
	~CNet(void);
public:

	static bool		InitWinSockLib(BYTE btMinor = 2, BYTE btMajor = 2);//默认参数为（2,2）
	static void		UnInitWinSockLib(void);
	bool			GetLocalIP(char* szHostIP);
};

