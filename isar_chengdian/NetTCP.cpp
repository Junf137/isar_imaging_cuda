//#include "StdAfx.h"
#include "NetTCP.h"


CNetTCP::CNetTCP(void)
{
	m_sSock = INVALID_SOCKET;
	m_sSockAccept = INVALID_SOCKET;
	m_bInitSuc = false;
	m_hWnd = NULL;
	m_iRecvLen = 0;
	m_pcRecvBuf = new char[TCP_SOCKET_TRANSMAX];

	m_iTCPType = TCP_CLIENT;
}

CNetTCP::~CNetTCP(void)
{
	if (m_pcRecvBuf != nullptr)
	{
		delete[] m_pcRecvBuf;
		m_pcRecvBuf = nullptr;
	}
}

bool CNetTCP::ServerOpen(WORD wLocalPort, HWND hWnd, unsigned int uiMsg)
{
	//如果窗口句柄无效，则失败
	if (NULL == hWnd)
	{
		return false;
	}
	else
	{
		m_hWnd = hWnd;
	}

	m_sSock = ::socket(AF_INET, SOCK_STREAM, 0);
	if (m_sSock == INVALID_SOCKET)
	{
		return false;
	}

	struct sockaddr_in server_sockaddr;

	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_port = htons(wLocalPort);
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);

	if (bind(m_sSock, (struct sockaddr*)&server_sockaddr, sizeof(struct sockaddr)) == -1)
	{
		m_bInitSuc = false;
		return false;
	}
	//开启线程
	if (listen(m_sSock, 5) == -1)
	{
		return false;
	}
	//
	struct sockaddr_in client_sockaddr;
	int iLen = sizeof(struct sockaddr);

	m_sSockAccept = accept(m_sSock, (struct sockaddr*)&client_sockaddr, &iLen);

	if (m_sSockAccept == INVALID_SOCKET)
	{
		m_bInitSuc = false;
		return false;
	}
	int	buf_size = TCP_SOCKET_TRANSMAX;
	//设置发送缓冲区大小为64K，如果未设置，默认为8K
	if (::setsockopt(m_sSockAccept, SOL_SOCKET, SO_SNDBUF, (char*)&buf_size, sizeof(int)) == SOCKET_ERROR)
	{
		/*m_bInitSuc = false;*/
		::closesocket(m_sSockAccept);
		m_sSockAccept = INVALID_SOCKET;
		return false;
	}
	//设置接收缓冲区大小为64K，如果未设置，默认为8K
	if (::setsockopt(m_sSockAccept, SOL_SOCKET, SO_RCVBUF, (char*)&buf_size, sizeof(int)) == SOCKET_ERROR)
	{
		/*m_bInitSuc = false;*/
		::closesocket(m_sSockAccept);
		m_sSockAccept = INVALID_SOCKET;
		return false;
	}
	//通知套接字端口有请求事件 wMsg就是消息，对于可以读的消息感兴趣
	int iRet = ::WSAAsyncSelect(m_sSockAccept, m_hWnd, uiMsg, FD_READ);
	if (iRet == SOCKET_ERROR)
	{
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
		return false;
	}

	m_iTCPType = TCP_SERVER;
	m_bInitSuc = true;
	return true;
}

bool CNetTCP::ClientOpen(char* pcServerAddr, WORD wServerPort, HWND hWnd, unsigned int uiMsg)
{
	//如果窗口句柄无效，则失败
	if (NULL == hWnd)
	{
		return false;
	}
	else
	{
		m_hWnd = hWnd;
	}

	m_sSock = ::socket(AF_INET, SOCK_STREAM, 0);
	if (m_sSock == INVALID_SOCKET)
	{
		m_bInitSuc = false;
		return false;
	}

	int	buf_size = TCP_SOCKET_TRANSMAX;

	//设置发送缓冲区大小为64K，如果未设置，默认为8K
	if (::setsockopt(m_sSock, SOL_SOCKET, SO_SNDBUF, (char*)&buf_size, sizeof(int)) == SOCKET_ERROR)
	{
		/*m_bInitSuc = false;*/
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
		return false;
	}

	//设置接收缓冲区大小为64K，如果未设置，默认为8K
	if (::setsockopt(m_sSock, SOL_SOCKET, SO_RCVBUF, (char*)&buf_size, sizeof(int)) == SOCKET_ERROR)
	{
		/*m_bInitSuc = false;*/
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
		return false;
	}

	struct sockaddr_in server_sockaddr;

	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_port = htons(wServerPort);
	server_sockaddr.sin_addr.s_addr = inet_addr(pcServerAddr);

	if (connect(m_sSock, (struct sockaddr*)&server_sockaddr, sizeof(struct sockaddr)) < 0)
	{
		m_bInitSuc = false;
		return false;
	}

	//通知套接字端口有请求事件 wMsg就是消息，对于可以读的消息感兴趣
	int iRet = ::WSAAsyncSelect(m_sSock, m_hWnd, uiMsg, FD_READ);
	if (iRet == SOCKET_ERROR)
	{
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
		return false;
	}

	m_iTCPType = TCP_CLIENT;
	m_bInitSuc = true;
	return true;
}


bool CNetTCP::ServerOpen(WORD wLocalPort)
{
	printf("%d\n", wLocalPort);
	m_sSock = ::socket(AF_INET, SOCK_STREAM, 0);
	if (m_sSock == INVALID_SOCKET)
	{
		return false;
	}

	struct sockaddr_in server_sockaddr;

	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_port = htons(wLocalPort);
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);

	if (bind(m_sSock, (struct sockaddr*)&server_sockaddr, sizeof(struct sockaddr)) == -1)
	{
		m_bInitSuc = false;
		return false;
	}
	//开启线程
	if (listen(m_sSock, 5) == -1)
	{
		return false;
	}
	//
	struct sockaddr_in client_sockaddr;
	int iLen = sizeof(struct sockaddr);

	m_sSockAccept = accept(m_sSock, (struct sockaddr*)&client_sockaddr, &iLen);

	if (m_sSockAccept == INVALID_SOCKET)
	{
		m_bInitSuc = false;
		return false;
	}
	int	buf_size = TCP_SOCKET_TRANSMAX;
	//设置发送缓冲区大小为64K，如果未设置，默认为8K
	if (::setsockopt(m_sSockAccept, SOL_SOCKET, SO_SNDBUF, (char*)&buf_size, sizeof(int)) == SOCKET_ERROR)
	{
		/*m_bInitSuc = false;*/
		::closesocket(m_sSockAccept);
		m_sSockAccept = INVALID_SOCKET;
		return false;
	}
	//设置接收缓冲区大小为64K，如果未设置，默认为8K
	if (::setsockopt(m_sSockAccept, SOL_SOCKET, SO_RCVBUF, (char*)&buf_size, sizeof(int)) == SOCKET_ERROR)
	{
		/*m_bInitSuc = false;*/
		::closesocket(m_sSockAccept);
		m_sSockAccept = INVALID_SOCKET;
		return false;
	}

	m_iTCPType = TCP_SERVER;
	m_bInitSuc = true;
	return true;
}

bool CNetTCP::ClientOpen(char* pcServerAddr, WORD wServerPort)
{
	if (NULL == pcServerAddr)
	{
		return false;
	}

	m_sSock = ::socket(AF_INET, SOCK_STREAM, 0);
	if (m_sSock == INVALID_SOCKET)
	{
		m_bInitSuc = false;
		return false;
	}

	int	buf_size = TCP_SOCKET_TRANSMAX;

	//设置发送缓冲区大小为64K，如果未设置，默认为8K
	if (::setsockopt(m_sSock, SOL_SOCKET, SO_SNDBUF, (char*)&buf_size, sizeof(int)) == SOCKET_ERROR)
	{
		/*m_bInitSuc = false;*/
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
		return false;
	}

	//设置接收缓冲区大小为64K，如果未设置，默认为8K
	if (::setsockopt(m_sSock, SOL_SOCKET, SO_RCVBUF, (char*)&buf_size, sizeof(int)) == SOCKET_ERROR)
	{
		/*m_bInitSuc = false;*/
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
		return false;
	}

	struct sockaddr_in server_sockaddr;

	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_port = htons(wServerPort);
	server_sockaddr.sin_addr.s_addr = inet_addr(pcServerAddr);

	if (connect(m_sSock, (struct sockaddr*)&server_sockaddr, sizeof(struct sockaddr)) < 0)
	{
		m_bInitSuc = false;
		return false;
	}

	m_iTCPType = TCP_CLIENT;
	m_bInitSuc = true;
	return true;
}

void CNetTCP::Close()
{
	if (m_iTCPType == TCP_CLIENT)
	{
		//不再接收套接字端口有请求消息
		int iRet = ::WSAAsyncSelect(m_sSock, m_hWnd, 0, 0);
		if (iRet == SOCKET_ERROR)
		{
			m_bInitSuc = false;
			::closesocket(m_sSock);
			m_sSock = INVALID_SOCKET;
		}
	}

	if (m_iTCPType == TCP_SERVER)
	{
		//不再接收套接字端口有请求消息
		int iRet = ::WSAAsyncSelect(m_sSockAccept, m_hWnd, 0, 0);
		if (iRet == SOCKET_ERROR)
		{
			m_bInitSuc = false;
			::closesocket(m_sSockAccept);
			m_sSockAccept = INVALID_SOCKET;
		}
	}

	m_bInitSuc = false;

	if (m_sSock != INVALID_SOCKET)
	{
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
	}

	if (m_sSockAccept != INVALID_SOCKET)
	{
		::closesocket(m_sSockAccept);
		m_sSockAccept = INVALID_SOCKET;
	}
}

/************************************************************************

************************************************************************/
bool CNetTCP::TransMsg(WPARAM wParam, LPARAM lParam)
{
	//检查是否是该套接字信息，以及网络是否发生了错误
	if (m_iTCPType == TCP_CLIENT)
	{
		if (m_sSock != (SOCKET)wParam || WSAGETSELECTERROR(lParam))
		{
			return false;
		}
	}
	else if (m_iTCPType == TCP_SERVER)
	{
		if (m_sSockAccept != (SOCKET)wParam || WSAGETSELECTERROR(lParam))
		{
			return false;
		}
	}
	else
	{
		return false;
	}

	//获取网络消息通知码
	WORD wMsg = WSAGETSELECTEVENT(lParam);

	if (wMsg == FD_READ)
	{
		return true;
	}
	else
	{
		return false;
	}
}
int CNetTCP::RecvData()
{
	if (m_iTCPType == TCP_CLIENT)
	{
		m_iRecvLen = recv(m_sSock, m_pcRecvBuf, TCP_SOCKET_TRANSMAX, 0);
		return m_iRecvLen;
	}
	else if (m_iTCPType == TCP_SERVER)
	{
		m_iRecvLen = recv(m_sSockAccept, m_pcRecvBuf, TCP_SOCKET_TRANSMAX, 0);
		return m_iRecvLen;
	}
	else
	{
		return -1;
	}
}

int CNetTCP::SendData(char* pcSendData, int iSendLen)
{
	int iRet = 0;

	if (m_iTCPType == TCP_CLIENT)
	{
		iRet = send(m_sSock, pcSendData, iSendLen, 0);
	}
	else if (m_iTCPType == TCP_SERVER)
	{
		iRet = send(m_sSockAccept, pcSendData, iSendLen, 0);
	}
	else
	{
		iRet = -1;
	}

	return iRet;
}
/************************************************************************
得到网络接受地址
************************************************************************/
char* CNetTCP::GetRecvBuf()
{
	return m_pcRecvBuf;
}
/************************************************************************
得到网络接受数据
************************************************************************/
int CNetTCP::GetRecvLen()
{
	return m_iRecvLen;
}