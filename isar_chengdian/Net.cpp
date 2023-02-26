//#include "StdAfx.h"
#include "Net.h"

CNet::CNet(void)
{
}

CNet::~CNet(void)
{

}
/**********************************************************
WinSock库初始化
**********************************************************/
bool CNet::InitWinSockLib(BYTE btMinor, BYTE btMajor)//默认参数为（2,2）
{
	bool bOpen = false;
	WSADATA wsaData;
	WORD sockVersion = MAKEWORD(btMinor, btMajor);
	int iRet = ::WSAStartup(sockVersion, &wsaData);
	if (0 == iRet)
	{
		bOpen = true;
	}
	return bOpen;
}
/**********************************************************
WinSock库卸载
**********************************************************/
void CNet::UnInitWinSockLib(void)
{
	::WSACleanup();
}
/************************************************************************
 获取本机IP地址
************************************************************************/
bool CNet::GetLocalIP(char* szHostIP)
{
	// 	WSADATA WSAData;
	// 	if(WSAStartup(MAKEWORD(2,0),&WSAData) != 0)
	// 	{
	// 		return false;
	// 	}

	char szName[20];
	gethostname(szName, sizeof(szName));
	struct hostent* pHostEnt = gethostbyname(szName);
	if (pHostEnt == NULL)
	{
		return false;
	}

	int n = 0;
	while (pHostEnt->h_addr_list[n] != NULL)
	{
		sprintf(szHostIP, ("%d.%d.%d.%d\n"),
			(pHostEnt->h_addr_list[n][0] & 0x00ff),
			(pHostEnt->h_addr_list[n][1] & 0x00ff),
			(pHostEnt->h_addr_list[n][2] & 0x00ff),
			(pHostEnt->h_addr_list[n][3] & 0x00ff));
		n++;
	}

	/*	WSACleanup();*/

	return true;
}


