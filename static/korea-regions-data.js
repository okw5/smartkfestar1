const koreaRegionsData = {
    // page.tsx의 `regions`에 해당
    regions: [
        { value: 'seoul', label: '서울특별시' },
        { value: 'busan', label: '부산광역시' },
        { value: 'incheon', label: '인천광역시' },
        { value: 'daegu', label: '대구광역시' },
        { value: 'gwangju', label: '광주광역시' },
        { value: 'daejeon', label: '대전광역시' },
        { value: 'ulsan', label: '울산광역시' },
        { value: 'sejong', label: '세종특별자치시' },
        { value: 'gyeonggi', label: '경기도' },
        { value: 'gangwon', label: '강원도' },
        { value: 'chungbuk', label: '충청북도' },
        { value: 'chungnam', label: '충청남도' },
        { value: 'jeonbuk', label: '전라북도' },
        { value: 'jeonnam', label: '전라남도' },
        { value: 'gyeongbuk', label: '경상북도' },
        { value: 'gyeongnam', label: '경상남도' },
        { value: 'jeju', label: '제주특별자치도' },
    ],

    // page.tsx의 `festivalTypes`에 해당
    festivalTypes: [
        '문화예술', '전통역사', '생태자연', '특산물', '주민화합', '관광', '체험행사', '기타'
    ],
    
    // page.tsx의 `getMunicipalitiesForRegion` 로직에 필요한 데이터
    municipalities: {
        seoul: [
            { value: 'gangnam', label: '강남구' }, { value: 'gangdong', label: '강동구' },
            { value: 'gangbuk', label: '강북구' }, { value: 'gangseo', label: '강서구' },
            { value: 'gwanak', label: '관악구' }, { value: 'gwangjin', label: '광진구' },
            { value: 'guro', label: '구로구' }, { value: 'geumcheon', label: '금천구' },
            { value: 'nowon', label: '노원구' }, { value: 'dobong', label: '도봉구' },
            { value: 'dongdaemun', label: '동대문구' }, { value: 'dongjak', label: '동작구' },
            { value: 'mapo', label: '마포구' }, { value: 'seodaemun', label: '서대문구' },
            { value: 'seocho', label: '서초구' }, { value: 'seongdong', label: '성동구' },
            { value: 'seongbuk', label: '성북구' }, { value: 'songpa', label: '송파구' },
            { value: 'yangcheon', label: '양천구' }, { value: 'yeongdeungpo', label: '영등포구' },
            { value: 'yongsan', label: '용산구' }, { value: 'eunpyeong', label: '은평구' },
            { value: 'jongno', label: '종로구' }, { value: 'jung', label: '중구' },
            { value: 'jungnang', label: '중랑구' }
        ],
        gyeonggi: [
            { value: 'suwon', label: '수원시' }, { value: 'seongnam', label: '성남시' },
            { value: 'yongin', label: '용인시' }, { value: 'goyang', label: '고양시' },
            { value: 'bucheon', label: '부천시' }, { value: 'ansan', label: '안산시' },
            { value: 'anyang', label: '안양시' }, { value: 'namyangju', label: '남양주시' },
            // ... 다른 경기도 시/군 추가
        ],
        // ... 다른 광역자치단체에 대한 데이터 추가
    },
	 dongs: {
        // 서울특별시 종로구의 읍/면/동
        jongno: [
            '청운효자동', '사직동', '삼청동', '부암동', '평창동', '무악동', 
            '교남동', '가회동', '종로1·2·3·4가동', '종로5·6가동', '이화동', 
            '혜화동', '창신1동', '창신2동', '창신3동', '숭인1동', '숭인2동'
        ],
        // 경기도 수원시의 읍/면/동 (일부)
        suwon: [
            '파장동', '정자1동', '정자2동', '정자3동', '영화동', '송죽동',
            '조원1동', '조원2동', '연무동', '세류1동', '세류2동', '세류3동',
            '평동', '서둔동', '구운동', '금곡동', '호매실동', '곡선동',
            '권선1동', '권선2동', '입북동', '영통1동', '영통2동', '영통3동',
            '망포1동', '망포2동', '매탄1동', '매탄2동', '매탄3동', '매탄4동',
            '원천동', '광교1동', '광교2동'
        ]
        // 필요한 다른 시/군/구 데이터를 여기에 추가할 수 있습니다.
        // 예: gangnam: ['신사동', '논현1동', '논현2동', ... ]
    }
    // =======================================================
};