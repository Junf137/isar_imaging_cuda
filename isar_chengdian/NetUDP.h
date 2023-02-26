#pragma once
#include "Net.h"

#define UDP_SOCKET_TRANSMAX 64*1024 //64K

/*#define UDP_SOCKET_TRANSMAX 32*1024 //64K*/

class CNetUDP :
	public CNet
{
public:
	CNetUDP(void);
	~CNetUDP(void);
public:
	SOCKET			m_sSock;
	HWND			m_hWnd;
	bool			m_bInitSuc;
	int				m_iRecvLen;
	char* m_pcRecvBuf;
	struct sockaddr_in m_addrRcvFrom;//收到的网络地址

	bool			JoinGroup(DWORD dwLocalAddress, DWORD dwGroupAddress);//加入组播
	bool			JoinGroup(DWORD dwLocalAddress, char* pcGroupAddress);//加入组播
	bool			JoinGroup(char* pcLocalAddress, DWORD dwGroupAddress);//加入组播
	bool			JoinGroup(char* pcLocalAddress, char* pcGroupAddress);//加入组播
	bool			JoinGroup(char* pcLocalAddress, char* pcGroupAddress, HWND hWnd, unsigned int uiMsg);//退出组

	bool			JropGroup(DWORD dwLocalAddress, DWORD dwGroupAddress);//退出组
	bool			JropGroup(DWORD dwLocalAddress, char* GroupAddress);//退出组
	bool			JropGroup(char* LocalAddress, DWORD dwGroupAddress);//退出组
	bool			JropGroup(char* LocalAddress, char* GroupAddress);//退出组


	bool			Open(char* pcAddress, WORD wPort);
	bool			Open(char* pcAddress, WORD wPort, HWND hWnd, unsigned int uiMsg);
	void			Close();

	int				SendData(char* pcBuffer, int iLength, char* pcAddress, WORD wPort);
	int				RcvData(char* pcBuffer, DWORD* pdwAddress, WORD* pwPort);
	int				RcvData();

	char* GetRecvBuf();
	UINT			GetRecvLen();

	bool			TransMsg(WPARAM wParam, LPARAM lParam);
};

