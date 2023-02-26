//#include "StdAfx.h"
#include "NetUDP.h"


CNetUDP::CNetUDP(void)
{
	m_sSock = INVALID_SOCKET;
	m_bInitSuc = false;
	m_hWnd = NULL;
	m_iRecvLen = 0;
	m_pcRecvBuf = new char[UDP_SOCKET_TRANSMAX];
}


CNetUDP::~CNetUDP(void)
{
	if (m_pcRecvBuf != nullptr)
	{
		delete[] m_pcRecvBuf;
		m_pcRecvBuf = nullptr;
	}
}


/**********************************************************
说明：加入组播
参数：dwLocalAddress		已加入组的本地IP地址，给出主机顺序的地址
	  dwGroupAddress		要离开的组播地址，给出主机顺序的地址
返回值：返回BOOL,成功为TRUE, 否则为FALSE
**********************************************************/
bool CNetUDP::JoinGroup(DWORD dwLocalAddress, DWORD dwGroupAddress)
{
	//设置多播组结构
	ip_mreq mreq;
	mreq.imr_interface.S_un.S_addr = htonl(dwLocalAddress); //本地地址
	mreq.imr_multiaddr.S_un.S_addr = htonl(dwGroupAddress); //多播地址 
	//加入一个多播组 
	if (setsockopt(m_sSock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) == SOCKET_ERROR)
	{
		return false;
	}
	else
	{
		return true;
	}
}

bool CNetUDP::JoinGroup(DWORD dwLocalAddress, char* pcGroupAddress)
{
	//设置多播组结构
	ip_mreq mreq;
	mreq.imr_interface.S_un.S_addr = htonl(dwLocalAddress); //本地地址
	mreq.imr_multiaddr.S_un.S_addr = inet_addr(pcGroupAddress); //多播地址 
	//加入一个多播组 
	if (setsockopt(m_sSock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) == SOCKET_ERROR)
	{
		return false;
	}
	else
	{
		return true;
	}
}

bool CNetUDP::JoinGroup(char* pcLocalAddress, DWORD dwGroupAddress)
{
	//设置多播组结构
	ip_mreq mreq;
	mreq.imr_interface.S_un.S_addr = inet_addr(pcLocalAddress); //本地地址
	mreq.imr_multiaddr.S_un.S_addr = htonl(dwGroupAddress); //多播地址 
	//加入一个多播组 
	if (setsockopt(m_sSock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) == SOCKET_ERROR)
	{
		return false;
	}
	else
	{
		return true;
	}
}

bool CNetUDP::JoinGroup(char* pcLocalAddress, char* pcGroupAddress)
{
	//设置多播组结构
	ip_mreq mreq;
	mreq.imr_interface.S_un.S_addr = inet_addr(pcLocalAddress); //本地地址
	mreq.imr_multiaddr.S_un.S_addr = inet_addr(pcGroupAddress); //多播地址 
	//加入一个多播组 
	if (setsockopt(m_sSock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) == SOCKET_ERROR)
	{
		return false;
	}
	else
	{
		return true;
	}
}



/**********************************************************
说明：离开组播
参数：	dwLocalAddress		已加入组的本地IP地址，给出主机顺序的地址
		dwGroupAddress		要离开的组播地址，给出主机顺序的地址
返回值：返回BOOL,成功为TRUE, 否则为FALSE
**********************************************************/
bool CNetUDP::JropGroup(DWORD dwLocalAddress, DWORD dwGroupAddress)
{
	//设置多播组结构
	ip_mreq mreq;
	mreq.imr_interface.S_un.S_addr = htonl(dwLocalAddress); //本地地址
	mreq.imr_multiaddr.S_un.S_addr = htonl(dwGroupAddress); //多播地址 
	//离开一个多播组
	if (setsockopt(m_sSock, IPPROTO_IP, IP_DROP_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) == SOCKET_ERROR)
	{
		return false;
	}
	else
	{
		return true;
	}
}

bool CNetUDP::JropGroup(DWORD dwLocalAddress, char* pcGroupAddress)
{
	//设置多播组结构
	ip_mreq mreq;
	mreq.imr_interface.S_un.S_addr = htonl(dwLocalAddress); //本地地址
	mreq.imr_multiaddr.S_un.S_addr = inet_addr(pcGroupAddress); //多播地址 
	//加入一个多播组 
	if (setsockopt(m_sSock, IPPROTO_IP, IP_DROP_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) == SOCKET_ERROR)
	{
		return false;
	}
	else
	{
		return true;
	}
}

bool CNetUDP::JropGroup(char* pcLocalAddress, DWORD dwGroupAddress)
{
	//设置多播组结构
	ip_mreq mreq;
	mreq.imr_interface.S_un.S_addr = inet_addr(pcLocalAddress); //本地地址
	mreq.imr_multiaddr.S_un.S_addr = htonl(dwGroupAddress); //多播地址 
	//加入一个多播组 
	if (setsockopt(m_sSock, IPPROTO_IP, IP_DROP_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) == SOCKET_ERROR)
	{
		return false;
	}
	else
	{
		return true;
	}
}

bool CNetUDP::JropGroup(char* pcLocalAddress, char* pcGroupAddress)
{
	//设置多播组结构
	ip_mreq mreq;
	mreq.imr_interface.S_un.S_addr = inet_addr(pcLocalAddress); //本地地址
	mreq.imr_multiaddr.S_un.S_addr = inet_addr(pcGroupAddress); //多播地址 
	//加入一个多播组 
	if (setsockopt(m_sSock, IPPROTO_IP, IP_DROP_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) == SOCKET_ERROR)
	{
		return false;
	}
	else
	{
		return true;
	}
}

/**********************************************************
打开网络
**********************************************************/
bool CNetUDP::Open(char* pcAddress, WORD wPort)
{
	//检查地址是否有效
	if (NULL == pcAddress)
	{
		return false;
	}
	//创建帧系列套接字
	m_sSock = ::socket(AF_INET, SOCK_DGRAM, 0);
	if (m_sSock == INVALID_SOCKET)
	{
		m_bInitSuc = false;
		return false;
	}

	struct sockaddr_in	addr;
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = inet_addr(pcAddress);
	addr.sin_port = htons(wPort);

	//将套接字绑定在指定的IP和端口上
	if (::bind(m_sSock, (LPSOCKADDR)&addr, sizeof(addr)) != 0)
	{
		/*m_bInitSuc = false;*/
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
		return false;
	}

	int	buf_size = UDP_SOCKET_TRANSMAX;

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

	m_bInitSuc = true;
	return true;
}
/**********************************************************
打开网络
**********************************************************/
bool CNetUDP::Open(char* pcAddress, WORD wPort, HWND hWnd, unsigned int uiMsg)
{
	//检查地址是否有效
	if (NULL == pcAddress)
	{
		return false;
	}
	//如果窗口句柄无效，则失败
	if (NULL == hWnd)
	{
		return false;
	}
	else
	{
		m_hWnd = hWnd;
	}
	//创建帧系列套接字
	m_sSock = ::socket(AF_INET, SOCK_DGRAM, 0);
	if (m_sSock == INVALID_SOCKET)
	{
		m_bInitSuc = false;
		return false;
	}

	struct sockaddr_in	addr;
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = inet_addr(pcAddress);
	addr.sin_port = htons(wPort);

	//将套接字绑定在指定的IP和端口上
	if (::bind(m_sSock, (LPSOCKADDR)&addr, sizeof(addr)) != 0)
	{
		/*m_bInitSuc = false;*/
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
		return false;
	}

	int	buf_size = UDP_SOCKET_TRANSMAX;

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

	//通知套接字端口有请求事件 wMsg就是消息，对于可以读的消息感兴趣
	int iRet = ::WSAAsyncSelect(m_sSock, m_hWnd, uiMsg, FD_READ);
	if (iRet == SOCKET_ERROR)
	{
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
		return false;
	}

	m_bInitSuc = true;
	return true;
}
/**********************************************************
关闭网络
**********************************************************/
void CNetUDP::Close()
{
	//不再接收套接字端口有请求消息
	int iRet = ::WSAAsyncSelect(m_sSock, m_hWnd, 0, 0);
	if (iRet == SOCKET_ERROR)
	{
		m_bInitSuc = false;
		::closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
	}
	//关闭
	if (m_sSock != INVALID_SOCKET)
	{
		m_bInitSuc = false;
		closesocket(m_sSock);
		m_sSock = INVALID_SOCKET;
	}
}

/************************************************************************
 发送数据
************************************************************************/
int	CNetUDP::SendData(char* pcBuffer, int iLength, char* pcAddress, WORD wPort)
{
	struct sockaddr_in	addr;
	memset(&addr, 0, sizeof(addr));

	addr.sin_family = AF_INET;
	addr.sin_port = htons(wPort);
	addr.sin_addr.s_addr = inet_addr(pcAddress);

	int iRet = ::sendto(m_sSock, pcBuffer, iLength, 0, (LPSOCKADDR)&addr, sizeof(addr));

	return iRet;
}

/************************************************************************
接收数据
************************************************************************/
int CNetUDP::RcvData(char* pcBuffer, DWORD* pdwAddress, WORD* pwPort)
{
	//检查变量是否合适
	if (pcBuffer == NULL || pdwAddress == NULL || pwPort == NULL)
	{
		return SOCKET_ERROR;
	}

	//读取数据
	struct sockaddr_in	addr;
	int iAddrSize = sizeof(addr);
	//非阻塞函数
	int iSize = ::recvfrom(m_sSock, pcBuffer, UDP_SOCKET_TRANSMAX - 1, 0, (LPSOCKADDR)&addr, &iAddrSize);

	*pdwAddress = addr.sin_addr.s_addr;
	*pwPort = addr.sin_port;

	*pdwAddress = htonl(*pdwAddress);
	*pwPort = htons(*pwPort);

	return iSize;
}

/************************************************************************
接收数据
************************************************************************/
int CNetUDP::RcvData()
{
	//检查变量是否合适
	//如果接收Buffer为空，则为其分配空间
	if (m_pcRecvBuf == NULL)
	{
		m_pcRecvBuf = new char[UDP_SOCKET_TRANSMAX];
	}
	//读取数据
	int iAddrSize = sizeof(m_addrRcvFrom);
	m_iRecvLen = ::recvfrom(m_sSock, m_pcRecvBuf, UDP_SOCKET_TRANSMAX - 1, 0, (LPSOCKADDR)&m_addrRcvFrom, &iAddrSize);

	return m_iRecvLen;
}
/************************************************************************
得到网络接受地址
************************************************************************/
char* CNetUDP::GetRecvBuf()
{
	return m_pcRecvBuf;
}
/************************************************************************
得到网络接受数据
************************************************************************/
UINT CNetUDP::GetRecvLen()
{
	return m_iRecvLen;
}
/************************************************************************

************************************************************************/
bool CNetUDP::TransMsg(WPARAM wParam, LPARAM lParam)
{
	//检查是否是该套接字信息，以及网络是否发生了错误
	if (m_sSock != (SOCKET)wParam || WSAGETSELECTERROR(lParam))
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