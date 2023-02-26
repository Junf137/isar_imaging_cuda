#pragma once
#include "Net.h"

#define TCP_SOCKET_TRANSMAX 64*1024 //64K

enum TCP_TYPE
{
	TCP_SERVER = 0,
	TCP_CLIENT
};

class CNetTCP :
	public CNet
{
public:
	CNetTCP(void);
	~CNetTCP(void);
public:
	SOCKET			m_sSock;
	HWND			m_hWnd;
	SOCKET			m_sSockAccept;
	bool			m_bInitSuc;
	char* m_pcRecvBuf;
	int				m_iRecvLen;
	int				m_iTCPType;

	bool			ServerOpen(WORD wLocalPort, HWND hWnd, unsigned int uiMsg);
	bool			ClientOpen(char* pcServerAddr, WORD wServerPort, HWND hWnd, unsigned int uiMsg);

	bool			ServerOpen(WORD wLocalPort);
	bool			ClientOpen(char* pcServerAddr, WORD wServerPort);

	void			Close();
	void			ClientClose();

	bool			TransMsg(WPARAM wParam, LPARAM lParam);

	int				SendData(char* pcSendData, int iSendLen);
	int				RecvData();

	int				ClientSend();
	int				ClientRecv();

	char* GetRecvBuf();
	int				GetRecvLen();
};

